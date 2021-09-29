

import os

OUTPUT_DIR ='/home/rizwan/DPR_models/biencoder_models_conala/'

CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_conala/dpr_biencoder.29.977"
pretrained_model="/home/rizwan/graphcodebert-base/"


top_k=100

qa_file_suffix = ['train', 'valid', "test"][2]

if qa_file_suffix!='test': FILE = "/home/rizwan/DPR_models/conala-corpus/conala-combined-train.json"
else: FILE = "/home/rizwan/DPR_models/conala-corpus/conala-test.json"
CTX_FILE = "/home/rizwan/DPR_models/conala-corpus/conala-mined.jsonl"
OUTPUT_ENCODED_FILE = "/local/rizwan/DPR_models/github_encoddings_conala-mined.jsonl_0.pkl"

DEVICES = [5]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python ' + \
          ' dense_retriever.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ CTX_FILE + \
          '  --qa_file '+ FILE + \
          '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILE + \
          '  --out_file  '+ OUTPUT_DIR+str(qa_file_suffix)+"_"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --batch_size 64 --match exact --sequence_length 256 --save_or_load_index --dataset conala_'+qa_file_suffix


# print (command)
print(command, flush=True)
os.system(command)








