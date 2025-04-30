# Surrogate Model Training

The `Coarse-grained_training_pre.py` demonstrates the process of obtaining top-k data. The training dataset for coarse-grained training consists of these top-k data combined with randomly sampled negative examples.

For the coarse-grained training stage, we utilize the Tevatron library with the following commands:

```sh
# Query embedding
python -m tevatron.retriever.driver.encode \
  --output_dir output \
  --model_name_or_path ../models/contriever \
  --fp16 \
  --encode_is_query \
  --per_device_eval_batch_size 256 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_path ../datasets/MS-MARCO_Passage_Ranking/ungz/dev.jsonl \
  --encode_output_path ../share/query.pt


# Generate corpus embeddings
python -m tevatron.retriever.driver.encode \
  --output_dir output \
  --model_name_or_path ../models/contriever \
  --fp16 \
  --per_device_eval_batch_size 32 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_path ../datasets/MS-MARCO_Passage_Ranking/ungz/corpus.jsonl \
  --encode_output_path ../share/corpus.pt

  
# Retrieval
python -m tevatron.retriever.driver.search \
  --query_reps ../share/query.pt \
  --passage_reps ../share/corpus.pt \
  --depth 10 \
  --batch_size 16 \
  --save_text \
  --save_ranking_to rank.tsv

python -m tevatron.utils.format.convert_result_to_marco \
              --input rank.tsv \
              --output rank.tsv.marco


# Training
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 -m tevatron.retriever.driver.train \
  --output_dir output \
  --model_name_or_path models/bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_path training_data_con.jsonl \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_group_size 10 \
  --dataloader_num_workers 1 \
  --learning_rate 1e-6 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --num_train_epochs 20 \
  --logging_steps 500 \
  --overwrite_output_dir
```

For the fine-grained training stage:
- Dataset preparation is detailed in `Fine-grained_training_pre.py`
- Training implementation can be found in `Fine-grained_training.py`
