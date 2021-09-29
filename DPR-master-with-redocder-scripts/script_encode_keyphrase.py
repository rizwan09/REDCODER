import os

lang='java'
CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_with_concode_tokens/KP20k/dpr_biencoder.9.15932"
pretrained_model="bert-base-uncased"

FILE = "/home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.test.jsonl"
OUTPUT_ENCODED_FILE = "/local/rizwan/DPR_models/github_encoddings_KP20k/KP20k.test.jsonl"

DEVICES = [0,1,2,3]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES + \
          ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
          ' generate_dense_embeddings.py ' \
          ' --model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_bert ' \
          ' --pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 128 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          ' --dataset KP20k ' \
          ' --num_shards 1 ' \
          ' --out_file ' + OUTPUT_ENCODED_FILE

print(command, flush=True)
os.system(command)








