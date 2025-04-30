import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast, GradScaler
import faiss
from typing import List, Dict
import json
import csv
import random
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import time

tokenizersg = AutoTokenizer.from_pretrained("models/contriever-sg")
modelsg = AutoModel.from_pretrained("models/contriever-sg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    modelsg = torch.nn.DataParallel(modelsg)
modelsg.to(device)
modelsg.eval()
scaler = GradScaler()

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def encodesg(texts):
    # Tokenize sentences
    encoded_input = tokenizersg(texts, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings with mixed precision
    with torch.no_grad(), autocast():
        model_output = modelsg(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output)
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # Convert to numpy array
    embeddings = embeddings.cpu().numpy()
    
    return embeddings


tokenizerori = AutoTokenizer.from_pretrained("models/contriever")
modelori = AutoModel.from_pretrained("models/contriever")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    modelori = torch.nn.DataParallel(modelori)
modelori.to(device)
modelori.eval()
scaler = GradScaler()

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def encodeori(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize sentences
    encoded_input = tokenizerori(
        texts, 
        padding=True, 
        max_length=512, 
        truncation=True, 
        return_tensors='pt'
    ).to(device)

    # Compute token embeddings with mixed precision
    with torch.no_grad(), autocast():
        outputs = modelori(**encoded_input, return_dict=True)
        
        embeddings = mean_pooling(
            outputs.last_hidden_state,
            encoded_input['attention_mask']
        )
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    
    return embeddings.cpu().numpy()


doc_content_dict = {}
with open("corpus.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line.strip())
        doc_content_dict[str(doc['docid'])] = doc['text']

ori_index = faiss.read_index("../share/con_nqa.index")
sg_index = faiss.read_index("../share/con_nqa_sg.index")


queries = []
with open("nfqa.csv", 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f, delimiter=',')
    next(csv_reader)  
    queries = [row[0] for row in csv_reader]

total_encode_time = 0
total_search_time = 0

with open("train-nqa-con.jsonl", 'w', encoding='utf-8') as out_f:
    for query in tqdm(queries, desc="processing"):
        
        start_time = time.time()
        query_vec_ori = encodeori(query).reshape(1, -1)
        encode_time = time.time() - start_time
        
        start_time = time.time()
        D_ori, I_ori = ori_index.search(query_vec_ori, k=3)
        search_time = time.time() - start_time
        
        
        start_time = time.time()
        query_vec_sg = encodesg(query).reshape(1, -1)
        encode_time += time.time() - start_time
        
        start_time = time.time()
        D_sg, I_sg = sg_index.search(query_vec_sg, k=20)
        search_time += time.time() - start_time
        
        
        top3_docs = [
            {"docid": str(idx), "text": doc_content_dict[str(idx)]}
            for idx in I_ori[0]
        ]
        
        top20_docs = [
            {"docid": str(idx), "text": doc_content_dict[str(idx)]}
            for idx in I_sg[0]
        ]
        
        
        top3_ids = set(doc['docid'] for doc in top3_docs)
        remaining_docs = [
            doc for doc in top20_docs 
            if doc['docid'] not in top3_ids
        ]
        
        
        available_ids = [did for did in doc_content_dict.keys() if did not in top3_ids]
        neg_docs = [
            {"docid": did, "text": doc_content_dict[did]}
            for did in random.sample(available_ids, 10)
        ]
        
        
        for rem_doc in remaining_docs:
            line_data = {
                'query': query,
                'doc1': top3_docs[0],
                'doc2': top3_docs[1],
                'doc3': top3_docs[2],
                'remaining_doc': rem_doc,
                'negative_docs': neg_docs
            }
            out_f.write(json.dumps(line_data, ensure_ascii=False) + '\n')
        
        total_encode_time += encode_time
        total_search_time += search_time
