import os

CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_with_concode_tokens/KP20k/dpr_biencoder.9.15932"

OUTPUT_DIR ='/home/rizwan/DPR_models/biencoder_models_with_concode_tokens/KP20k/'

pretrained_model="bert-base-uncased"

top_k=20000
qa_file_suffix = "test"
CTX_FILE = "/home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.test.jsonl"
FILE = "/home/wasiahmad/workspace/projects/NeuralKpGen/retrieval/results/KP20k.test.bart.jsonl"
OUTPUT_ENCODED_FILE = "/local/rizwan/DPR_models/github_encoddings_KP20k/KP20k.test.jsonl_0.pkl"

DEVICES = [5]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python ' + \
          ' dense_retriever.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ CTX_FILE + \
          '  --qa_file '+ FILE + \
          '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILE + \
          '  --out_file  '+ OUTPUT_DIR+str(qa_file_suffix)+"_"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --batch_size 64 --match exact --sequence_length 256 --save_or_load_index --dataset KP20k '



# print (command)
print(command, flush=True)
os.system(command)








