import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import logging
import os
import random
import numpy as np
import shutil
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HierarchicalDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"loading data: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="loading data"):
                self.samples.append(json.loads(line))
                
        logger.info(f"loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        query_encoding = self.tokenizer(
            sample['query'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        doc1_encoding = self.tokenizer(
            sample['doc1']['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        doc2_encoding = self.tokenizer(
            sample['doc2']['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        doc3_encoding = self.tokenizer(
            sample['doc3']['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        remaining_encoding = self.tokenizer(
            sample['remaining_doc']['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        neg_texts = [neg['text'] for neg in sample['negative_docs']]
        neg_encoding = self.tokenizer(
            neg_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'doc1_input_ids': doc1_encoding['input_ids'].squeeze(0),
            'doc1_attention_mask': doc1_encoding['attention_mask'].squeeze(0),
            'doc2_input_ids': doc2_encoding['input_ids'].squeeze(0),
            'doc2_attention_mask': doc2_encoding['attention_mask'].squeeze(0),
            'doc3_input_ids': doc3_encoding['input_ids'].squeeze(0),
            'doc3_attention_mask': doc3_encoding['attention_mask'].squeeze(0),
            'remaining_input_ids': remaining_encoding['input_ids'].squeeze(0),
            'remaining_attention_mask': remaining_encoding['attention_mask'].squeeze(0),
            'neg_input_ids': neg_encoding['input_ids'],
            'neg_attention_mask': neg_encoding['attention_mask']
        }

class DualEncoder(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        return outputs.last_hidden_state[:, 0]  # [CLS] token

class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.margin_doc1_2 = 0.2    # doc1 and doc2
        self.margin_doc2_3 = 0.1    # doc2 and doc3
        self.margin_doc3_remaining = 0.15  # doc3 and remaining
        self.margin_remaining_neg = 0.3    # remaining and negative

    def forward(self, query_emb, doc1_emb, doc2_emb, doc3_emb, remaining_emb, neg_embs):
        doc1_sim = torch.sum(query_emb * doc1_emb, dim=1)
        doc2_sim = torch.sum(query_emb * doc2_emb, dim=1)
        doc3_sim = torch.sum(query_emb * doc3_emb, dim=1)
        remaining_sim = torch.sum(query_emb * remaining_emb, dim=1)
        neg_sims = torch.sum(query_emb.unsqueeze(1) * neg_embs, dim=2)
        
        loss = torch.zeros_like(doc1_sim)
        
        loss += torch.relu(self.margin_doc1_2 - (doc1_sim - doc2_sim))
        loss += torch.relu(self.margin_doc2_3 - (doc2_sim - doc3_sim))
        loss += torch.relu(self.margin_doc3_remaining - (doc3_sim - remaining_sim))
        loss += torch.relu(self.margin_remaining_neg - (remaining_sim.unsqueeze(1) - neg_sims)).mean(dim=1)
        
        return loss.mean()

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    fh = logging.FileHansger(f"{args.output_dir}/training.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHansger(fh)
    
    logger.info(f"loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DualEncoder(args.model_name)
    model = model.to(args.device)
    
    train_dataset = HierarchicalDataset(args.train_file, tokenizer, args.max_length)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = PairwiseRankingLoss()
    
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_training_steps,
        eta_min=args.min_lr
    )
    
    logger.info("start training...")
    global_step = 0
    best_loss = float('inf')
    best_checkpoint_dir = None
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            query_emb = model(batch['query_input_ids'], batch['query_attention_mask'])
            doc1_emb = model(batch['doc1_input_ids'], batch['doc1_attention_mask'])
            doc2_emb = model(batch['doc2_input_ids'], batch['doc2_attention_mask'])
            doc3_emb = model(batch['doc3_input_ids'], batch['doc3_attention_mask'])
            remaining_emb = model(batch['remaining_input_ids'], batch['remaining_attention_mask'])
            
            batch_size, neg_count, seq_len = batch['neg_input_ids'].shape
            neg_input_ids = batch['neg_input_ids'].view(-1, seq_len)
            neg_attention_mask = batch['neg_attention_mask'].view(-1, seq_len)
            neg_embs = model(neg_input_ids, neg_attention_mask)
            neg_embs = neg_embs.view(batch_size, neg_count, -1)
            
            loss = criterion(query_emb, doc1_emb, doc2_emb, doc3_emb, remaining_emb, neg_embs)
            
            optimizer.zero_grad()
            loss.backward()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if global_step % args.log_steps == 0:
                with torch.no_grad():
                    doc1_sim = torch.sum(query_emb * doc1_emb, dim=1).mean()
                    doc2_sim = torch.sum(query_emb * doc2_emb, dim=1).mean()
                    doc3_sim = torch.sum(query_emb * doc3_emb, dim=1).mean()
                    remaining_sim = torch.sum(query_emb * remaining_emb, dim=1).mean()
                    neg_sim = torch.sum(query_emb.unsqueeze(1) * neg_embs, dim=2).mean()
                    logger.info(f"\nsimilarity statistics:")
                    logger.info(f"Doc1: {doc1_sim:.4f}")
                    logger.info(f"Doc2: {doc2_sim:.4f}")
                    logger.info(f"Doc3: {doc3_sim:.4f}")
                    logger.info(f"Remaining: {remaining_sim:.4f}")
                    logger.info(f"Negative: {neg_sim:.4f}")
            
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                avg_loss = epoch_loss / global_step
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.encoder.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    best_checkpoint_dir = checkpoint_dir
                    logger.info(f"save best checkpoint to: {checkpoint_dir}")
        
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} average loss: {avg_loss:.4f}")
    
    if best_checkpoint_dir is not None:
        final_dir = os.path.join(args.output_dir, 'final-model')
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(best_checkpoint_dir, final_dir)
        logger.info(f"copy best checkpoint to final-model: {final_dir}")
        
        with open(os.path.join(final_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("save training config")
    
    return model, tokenizer

if __name__ == "__main__":
    class Args:
        train_file = 'train-nqa-con.jsonl'
        model_name = 'models/contriever-sg'
        output_dir = 'hierarchical_contriever'
        max_length = 512
        batch_size = 4
        epochs = 20
        learning_rate = 2e-5
        min_lr = 1e-6
        weight_decay = 0.01
        save_steps = 1000
        log_steps = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = 42
        num_workers = 4
        max_grad_norm = 1.0
    
    args = Args()
    train(args)