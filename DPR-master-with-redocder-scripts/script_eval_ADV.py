import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]

lang='python'
# lang='java'

CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_codexglue_text_code/python/dpr_biencoder.8.10493"

pretrained_model="microsoft/codebert-base"


OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_codexglue_text_code/"+lang



pretrained_model="microsoft/codebert-base"





OUTPUT_ENCODED_FILES = "/home/rizwan/DPR_models/github_encoddings/python/adv_encoddings_test.jsonl_0.pkl"
qa_file = "/home/rizwan/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset/test.jsonl"
print(qa_file)

DEVICES = [2]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

top_k =  19210
OUT_TOP_K_FILE = OUTPUT_DIR_PATH+CHECKPOINT.split()[-1]

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python dense_retriever.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ qa_file + \
          '  --qa_file '+ qa_file + \
          '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILES + \
          '  --out_file  '+ OUTPUT_DIR_PATH+"_ADV_test"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --sequence_length 256 --save_or_load_index --CSNET_ADV'
# print (command)
print(command, flush=True)
os.system(command)




