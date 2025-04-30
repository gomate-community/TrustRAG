import torch
import numpy as np
from tqdm import tqdm
import math
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt  
from tqdm import tqdm         
import datetime             
import transformers
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import faiss
from datetime import datetime
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import math            
import json                
import linecache
model_id = "../llama3-8b-instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)



def attack_env_func(doc):
    index_save_path = 'factqaindex/orifqa.index'  
    indexori = faiss.read_index(index_save_path)
    query_vectorori=encodeori([question])
    
    k = 3
    scores, ids = indexori.search(query_vectorori, k)
    contextvalue=[]
    for i in range(k):
        doc_id = ids[0][i]
        score = scores[0][i]
            
       
        line = linecache.getline('corpusfqa.jsonl', doc_id + 1)
        data = json.loads(line)
        text = data['text']
        contextvalue.append(text)
            
    linecache.clearcache()
        
    
    context=contextvalue
    context.insert(flag,doc)
    context=context[:3]
    messages = [
    {"role": "system", "content": f"You are a chatbot who Uses the following pieces of retrieved context to answer the question. Context: {context}."},
    {"role": "user", "content": f"Start by answering the question and then briefly exlpain why. Question:{question}"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
    		messages, 
    		tokenize=False, 
    		add_generation_prompt=True
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=False,  
        temperature=0.1,    
        top_p=0.9,         
        num_beams=3,     
        eos_token_id=terminators
    )
    text=outputs[0]["generated_text"][len(prompt):]
    
    
    messagesvalue = [
        {"role": "system", "content": f"You are a chatbot who directly performs the user's tasks"},
        {"role": "user", "content": f'''Given a discussion and a document on a query, you need to evaluate how well the discussion references the document.You need to carefully analyze the content of the discussion and the document and then giving an exact score between 0 and 1.You just need to output the score and don't output extra content!
                                    Query:{question}. Discussion:{text}.Document:{document}.'''},]
    prompt = pipeline.tokenizer.apply_chat_template(
    		messagesvalue, 
    		tokenize=False, 
    		add_generation_prompt=True
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputsvalue = pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=False,  
        temperature=0.1,    
        top_p=0.9,         
        num_beams=3,     
        eos_token_id=terminators
    )
    score=outputsvalue[0]["generated_text"][len(prompt):]
    score=float(score)
    return score

class SynonymReplacementOptimizer:
    def __init__(self, tokenizer, model, device, query_vector, query_words):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.query_vector = query_vector
        self.query_words = query_words
        
        
        self.epsilon = 0.2        
        self.gamma = 0.5         
        self.value_coef = 0.5     
        self.entropy_coef = 0.01  
        
        
        self.position_history = {}  
        self.position_stats = {}    
        
        
        self.state_encoder = nn.Sequential(
            nn.Linear(1545, 512),    
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128)
        ).to(device)
        
       
        self.candidate_encoder = nn.Sequential(
            nn.Linear(769, 256),    
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)
        
        
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),    
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        
        
        self.value_net = nn.Sequential(
            nn.Linear(1545, 512),    
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        
        
        self.optimizer = torch.optim.Adam([
            {'params': self.state_encoder.parameters()},
            {'params': self.candidate_encoder.parameters()},
            {'params': self.policy_head.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=0.0003)
        
    def _get_context(self, doc, position, window_size=2):
        
        tokens = doc.split()
        start = max(0, position - window_size)
        end = min(len(tokens), position + window_size + 1)
        return ' '.join(tokens[start:end])
    
    def _get_relative_position(self, position, doc_length):
        
        return {
            'relative_pos': position / doc_length,
            'is_start': position == 0,
            'is_end': position == doc_length - 1,
            'segment': position / (doc_length / 3)  
        }
    
    def _get_state(self, doc, position, history):
        
        tokens = doc.split()
        target_word = tokens[position]

        
        doc_vec = encode([doc])[0].flatten()  

        
        word_vec = encode([target_word])[0].flatten() 

        
        temp_tokens = tokens.copy()
        temp_tokens.pop(position)
        temp_doc = ' '.join(temp_tokens)

        original_emb = encode([doc])[0]
        temp_emb = encode([temp_doc])[0]

        original_score = np.dot(self.query_vector, original_emb.T)
        temp_score = np.dot(self.query_vector, temp_emb.T)
        importance_score = float(abs(original_score - temp_score))  

        
        relative_pos = float(position / len(tokens))  

       
        history_data = history.get(position, [])
        recent_rewards = [r for _, r in history_data[-5:]]
        while len(recent_rewards) < 5:
            recent_rewards.append(0)

        history_features = [
            float(min(len(history_data) / 10, 1.0)),  
            float(np.mean([r for _, r in history_data]) if history_data else 0),  
            *[float(r) for r in recent_rewards]  
        ]

        
        state = np.concatenate([
            doc_vec,              
            word_vec,             
            [importance_score],   
            [relative_pos],       
            history_features      
        ])  

        return state
    def _get_similar_tokens(self, token_id, query_embedding, top_k=10):
        
        top_k=20
        with torch.no_grad():
            token_embedding = self.model.embeddings.word_embeddings(
                torch.tensor([token_id]).to(self.device)
            )

        perturbed_emb = token_embedding.clone().detach()
        query_emb = torch.tensor(query_embedding, device=self.device)

        pgd_steps = 15        
        step_size = 0.02      
        epsilon = 0.2         

        for _ in range(pgd_steps):
            perturbed_emb.requires_grad_(True)

            query_similarity = torch.cosine_similarity(
                perturbed_emb, 
                query_emb.unsqueeze(0)
            )

            token_similarity = torch.cosine_similarity(
                perturbed_emb,
                token_embedding
            )

            similarity_threshold = 0.6  
            semantic_penalty = torch.relu(similarity_threshold - token_similarity)
            loss = -query_similarity + 0.3 * semantic_penalty  

            loss.backward()

            with torch.no_grad():
                grad = perturbed_emb.grad
                grad = grad / (torch.norm(grad) + 1e-8)
                perturbed_emb = perturbed_emb - step_size * grad

                delta = perturbed_emb - token_embedding
                delta = torch.clamp(delta, -epsilon, epsilon)
                perturbed_emb = token_embedding + delta

                perturbed_emb = perturbed_emb.detach()

        with torch.no_grad():
            vocab_embeddings = self.model.embeddings.word_embeddings.weight

            query_similarities = torch.cosine_similarity(
                query_emb.unsqueeze(0), 
                vocab_embeddings
            )
            token_similarities = torch.cosine_similarity(
                token_embedding, 
                vocab_embeddings
            )

            semantic_mask = token_similarities >= 0.6  

            query_similarities[~semantic_mask] = float('-inf')
            top_k_values, top_k_indices = torch.topk(query_similarities, min(top_k, semantic_mask.sum()))

        return top_k_indices.cpu().numpy()
    
    def _get_candidate_synonyms(self, word, top_k=10):
        
        candidates = []
        word_vec = encode([word])[0]
        
        query_candidates = []
        for query_word in self.query_words:
            query_word_vec = encode([query_word])[0]
            similarity = np.dot(word_vec, query_word_vec)
            if similarity > 0.7:  
                query_candidates.append((query_word, similarity * 1.1))  
        
        token_id = self.tokenizer.encode(word)[1]
        similar_tokens = self._get_similar_tokens(token_id, top_k)
        for token in similar_tokens:
            token_word = self.tokenizer.decode([token])
            if token_word != word:  
                token_vec = encode([token_word])[0]
                similarity = np.dot(word_vec, token_vec)
                candidates.append((token_word, similarity))
        
        candidates = query_candidates + candidates
        candidates = sorted(set(candidates), key=lambda x: x[1], reverse=True)
        
        final_candidates = []
        seen_words = set()
        
        final_candidates.append((word, 1.0))
        seen_words.add(word)
        
        for candidate, score in candidates:
            if candidate not in seen_words and len(final_candidates) < top_k:
                final_candidates.append((candidate, score))
                seen_words.add(candidate)
        
        while len(final_candidates) < top_k:
            final_candidates.append((word, 0.0))
        print(word)
        print(final_candidates)
        return final_candidates
    
    def _select_position(self, doc, history):
        
        tokens = doc.split()
        position_scores = {}
        
        original_emb = encode([doc])[0]
        original_score = np.dot(self.query_vector, original_emb.T)
        
        for pos in range(len(tokens)):
            temp_tokens = tokens.copy()
            temp_tokens.pop(pos)
            temp_doc = ' '.join(temp_tokens)
            
            temp_emb = encode([temp_doc])[0]
            temp_score = np.dot(self.query_vector, temp_emb.T)
            
            importance = abs(original_score - temp_score) * 100
            
            stats = self.position_stats.get(pos, {
                'success_rate': 0,
                'avg_reward': 0,
                'attempts': 0
            })
            
            history_score = (
                stats.get('success_rate', 0) * 0.35 +
                stats.get('avg_reward', 0) * 0.35 +
                (1 / (1 + stats.get('attempts', 0))) * 0.3 
            )
            
            position_scores[pos] = (
                importance * 0.3 +           
                history_score * 0.4 +        
                np.random.normal(0, 0.1) * 0.3 
            )
        
        return max(position_scores.items(), key=lambda x: x[1])[0]
    
    def _select_action(self, state, candidates):
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        state_features = self.state_encoder(state_tensor)
        
        candidate_scores = []
        for word, score in candidates:
            word_vec = encode([word])[0]
            word_vec = encode([word])[0]
            word_features = np.concatenate([word_vec, [score]])
            word_tensor = torch.FloatTensor(word_features).to(self.device)
            
            cand_features = self.candidate_encoder(word_tensor)
            
            combined = torch.cat([state_features, cand_features])
            score = self.policy_head(combined)
            candidate_scores.append(score)
        
        candidate_scores = torch.cat(candidate_scores)  
        action_probs = F.softmax(candidate_scores, dim=0)

            
        action = torch.multinomial(action_probs, 1).item()
        return candidates[action][0], action_probs[action].item()
    
    
    def _update_history(self, position, old_word, new_word, reward):
        
        if position not in self.position_history:
            self.position_history[position] = []
            
        if len(self.position_history[position]) >= 10:
            self.position_history[position].pop(0)
            
        self.position_history[position].append((
            {'old_word': old_word, 'new_word': new_word},
            reward
        ))
        
        if position not in self.position_stats:
            self.position_stats[position] = {
                'attempts': 0,
                'successes': 0,
                'total_reward': 0
            }
            
        stats = self.position_stats[position]
        stats['attempts'] += 1
        if reward > 0:
            stats['successes'] += 1
        stats['total_reward'] += reward
        stats['success_rate'] = stats['successes'] / stats['attempts']
        stats['avg_reward'] = stats['total_reward'] / stats['attempts']
    
    def _compute_position_based_returns(self, trajectories):
        returns = []
        for t in trajectories:
            position = t['position']
            position_history = self.position_history[position]
            current_idx = -1
            for i, (history_info, _) in enumerate(position_history):
                if history_info['new_word'] == t['candidates'][t['action']][0]:
                    current_idx = i
                    break
            if current_idx == -1:
                
                returns.append(t['reward'])
                continue
            future_return = 0
            start_idx = max(len(position_history)-3, current_idx)

            for i in range(len(position_history)-1, start_idx-1, -1):
                _, reward = position_history[i]
                future_return = reward + self.gamma * future_return

            returns.append(future_return)

        
        if not returns:
            return [0.0]  

        return returns
        
    def _update_policy(self, trajectories):
        states = torch.FloatTensor([t['state'] for t in trajectories]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in trajectories]).to(self.device)
        old_action_probs = torch.FloatTensor([t['action_prob'] for t in trajectories]).to(self.device)
        candidates_list = [t['candidates'] for t in trajectories]
        
        returns = self._compute_position_based_returns(trajectories)
        returns = torch.FloatTensor(returns).to(self.device)
        
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            advantages = returns - values  
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(10):
            total_policy_loss = 0
            
    
            for i in range(len(trajectories)):
                state = states[i]
                action_idx = actions[i]
                candidates = candidates_list[i]
                
        
                state_features = self.state_encoder(state)
                candidate_scores = []
                
                for word, score in candidates:
                    word_vec = encode([word])[0]
                    word_features = np.concatenate([word_vec, [score]])
                    word_tensor = torch.FloatTensor(word_features).to(self.device)
                    cand_features = self.candidate_encoder(word_tensor)
                    
                    combined = torch.cat([state_features, cand_features])
                    score = self.policy_head(combined)
                    candidate_scores.append(score)
                
                action_probs = F.softmax(torch.stack(candidate_scores), dim=0)
                new_action_prob = action_probs[action_idx]
                
               
                ratio = new_action_prob / old_action_probs[i]
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages[i]
                policy_loss = -torch.min(surr1, surr2)
                
                total_policy_loss += policy_loss
            
          
            value_pred = self.value_net(states).squeeze()
            value_loss = F.mse_loss(value_pred, returns)  
            
            
            loss = (total_policy_loss.mean() + 
                    self.value_coef * value_loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            
          
            for net in [self.state_encoder, self.candidate_encoder, 
                    self.policy_head, self.value_net]:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                
            self.optimizer.step()
            
            
            if torch.mean((new_action_prob - old_action_probs[i]).abs()) > 0.015:
                break
    
    def _replace_word(self, doc, position, new_word):
        
        tokens = doc.split()
        tokens[position] = new_word
        return ' '.join(tokens)
    
    def _compute_score(self, doc):
      
        doc_vec = encode([doc])[0]
        score = np.dot(doc_vec, self.query_vector.T)
        return float(score)  
    
    def optimize_document(self, document, target_score, max_iterations):
        current_doc = document
        current_score = float(self._compute_score(document))
        best_doc = document
        best_score = current_score
        semantic_threshold = 0.97
        trajectories = []
        improvement_threshold = 0.0005
        
       
        stats = {
            'iterations': 0,
            'successful_changes': 0,
            'total_attempts': 0,
            'modified_positions': set(), 
            'total_words': len(document.split()),  
            'position_changes': {},  
            'score_history': [],    
            'semantic_similarities': [],
            'final_modifications': {}  
        }
        
        for iteration in tqdm(range(max_iterations)):
            stats['iterations'] = iteration + 1
            stats['total_attempts'] += 1
            
            
            position = self._select_position(current_doc, self.position_history)
            
            
            current_word = current_doc.split()[position]
            candidates = self._get_candidate_synonyms(current_word)
            
            
            state = self._get_state(current_doc, position, self.position_history)
            new_word, action_prob = self._select_action(state, candidates)
            
            
            new_doc = self._replace_word(current_doc, position, new_word)
            new_score = float(self._compute_score(new_doc))
            score_improvement = new_score - current_score
            
           
            new_vec = encode([new_doc])[0]
            orig_vec = encode([document])[0]
            semantic_similarity = float(np.dot(orig_vec, new_vec.T))
            stats['semantic_similarities'].append(semantic_similarity)
            
            
            if (semantic_similarity >= semantic_threshold and 
                score_improvement > improvement_threshold):
                
                
                new_quality = attack_env_func(new_doc)
                current_quality = attack_env_func(current_doc)
                quality_change = new_quality - current_quality
                
                
                reward = self._compute_reward(
                    old_score=current_score,
                    new_score=new_score,
                    target_score=target_score,
                    semantic_similarity=semantic_similarity,
                    semantic_threshold=semantic_threshold,
                    quality_change=quality_change
                )
                
                
                if quality_change >= 0:
                    stats['successful_changes'] += 1
                    stats['modified_positions'].add(position)
                    current_doc = new_doc
                    current_score = new_score
                    if new_score > best_score:
                        best_doc = new_doc
                        best_score = new_score
                        print(f"New best score: {float(best_score):.4f} "
                              f"(improvement: {score_improvement:.4f}, "
                              f"quality: {new_quality:.4f})")
            else:
                
                reward = self._compute_reward(
                    old_score=current_score,
                    new_score=new_score,
                    target_score=target_score,
                    semantic_similarity=semantic_similarity,
                    semantic_threshold=semantic_threshold
                )
            
            
            stats['score_history'].append(new_score)
            
            
            if position not in stats['position_changes']:
                stats['position_changes'][position] = []
            stats['position_changes'][position].append({
                'iteration': iteration,
                'old_word': current_word,
                'new_word': new_word,
                'score_change': new_score - current_score,
                'semantic_similarity': semantic_similarity
            })
            
          
            trajectories.append({
                'state': state,
                'action': [c[0] for c in candidates].index(new_word),
                'reward': reward,
                'action_prob': action_prob,
                'candidates': candidates,
                'position':position
            })
            
            
            self._update_history(position, current_word, new_word, reward)
            
            
            if len(trajectories) >= 10:
                self._update_policy(trajectories)
                trajectories = []
            
            if current_score >= target_score:
                break
            vecdoc=encodeori([current_doc])
            scoredoc=np.dot(query_vectorori,vecdoc.T)
            if scoredoc>tarscore:
                print(scoredoc)
                break
                
        perturbation_rate = len(stats['modified_positions']) / stats['total_words']
        final_similarity = stats['semantic_similarities'][-1]
        return best_doc, best_score, stats, perturbation_rate, final_similarity

    def _compute_reward(self, old_score, new_score, target_score, semantic_similarity, semantic_threshold, quality_change=None):
        
        reward = 0
        
       
        if semantic_similarity < semantic_threshold:
            reward -= 2.0  
        
        
        score_diff = new_score - old_score
        if score_diff > 0:
            reward += score_diff * 100.0  
        else:
            reward += -0.2+score_diff * 100.0  
        
        
        if new_score >= target_score:
            reward += 5.0  
        
        
        if quality_change is not None:
            if quality_change >= 0:
                reward += quality_change * 2  
            else:
                reward += quality_change * 1  
                
        return reward
    
    
tokenizer = AutoTokenizer.from_pretrained("models/sg-fqa")
model = AutoModel.from_pretrained("models/sg-fqa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)
model.eval()


scaler = GradScaler()

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def encode(texts):
    encoded_input = tokenizer(texts, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad(), autocast():
        model_output = model(**encoded_input, return_dict=True)
    embeddings = cls_pooling(model_output)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().numpy()
    return embeddings


tokenizerori = AutoTokenizer.from_pretrained("models/co-condenser-marco-retriever")
modelori = AutoModel.from_pretrained("models/co-condenser-marco-retriever")
if torch.cuda.device_count() > 1:
    modelori = torch.nn.DataParallel(modelori)
modelori.to(device)
modelori.eval()
flag = 2

def encodeori(texts):
    encoded_input = tokenizerori(texts, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad(), autocast():
        model_output = modelori(**encoded_input, return_dict=True)
    embeddings = cls_pooling(model_output)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().numpy()
    return embeddings

def get_keywords_exact(query, dfk):
    result = dfk[dfk['Question'] == query]['keywords']
    keywords_str = result.iloc[0]
   
    keywords_list = eval(keywords_str)  
 
    return keywords_list

def cleanup_temp_files(pattern='RP_lma_temp_fqa_1*.csv'):
  
    temp_files = glob.glob(pattern)
    for file in temp_files:
        try:
            os.remove(file)
            print(f"Removed temporary file: {file}")
        except Exception as e:
            print(f"Error removing {file}: {str(e)}")




index_save_path_sg = 'factqaindex/sgfqa.index'
index_save_path_ori = 'factqaindex/orifqa.index'
indexsg = faiss.read_index(index_save_path_sg)
indexori = faiss.read_index(index_save_path_ori)



dfk = pd.read_csv('fqa.csv', delimiter=',')
results = []
k = 3


for index, row in dfk.iterrows():
    
    question = row['Question']
    print(f"Processing question {index + 1}/{len(dfk)}: {question}")
    query_vectorori = encodeori([question])
    document = row['attackdoc']
    
    query_vector = encode([question])
    query_words = get_keywords_exact(question, dfk)

    if not query_vector.flags['C_CONTIGUOUS']:
        query_vector = np.ascontiguousarray(query_vector)
    tscore, tid = indexsg.search(query_vector, 1)
    tscore = tscore.item()

    scores, ids = indexori.search(query_vectorori, k)
    tarscore = scores[0][flag]
    optimizer = SynonymReplacementOptimizer(tokenizer, model, device, query_vector, query_words)
    optimized_doc, final_score, stats, purtb, sim = optimizer.optimize_document(
        document=document,
        target_score=tscore,
        max_iterations=300
        )
        
