import os

lang='python'
CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_conala//dpr_biencoder.29.977"
pretrained_model="/home/rizwan/graphcodebert-base/"

FILE = "/home/rizwan/DPR_models/conala-corpus/conala-mined.jsonl"
# FILE = "/home/rizwan/DPR_models/external-knowledge-codegen/apidocs/processed/distsmpl/snippet_15k/unique_snippets.txt"
OUTPUT_ENCODED_FILE = "/local/rizwan/DPR_models/github_encoddings_conala-mined.jsonl"

DEVICES = [1,3]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES + \
          ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
          ' generate_dense_embeddings.py ' \
          ' --model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_bert ' \
          ' --pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 128 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          ' --dataset conala ' \
          ' --num_shards 1 ' \
          ' --out_file ' + OUTPUT_ENCODED_FILE

print(command, flush=True)
os.system(command)








