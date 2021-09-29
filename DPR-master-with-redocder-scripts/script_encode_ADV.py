import os

lang='python'


CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_codexglue_text_code/python/dpr_biencoder.8.10493"

GITHUB_DB="/home/rizwan/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset/"

OUTPUT_DIR ='/home/rizwan/DPR_models/github_encoddings/'+lang+"/"

pretrained_model="microsoft/codebert-base"



# for id in range (1): #(len(files)):
#     FILE =  GITHUB_DB+files[-(id+1)]
file_name='test.jsonl'
FILE =  GITHUB_DB+file_name
OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'adv_encoddings_'  +file_name

DEVICES = [2]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES + \
          ' python ' + \
          ' generate_dense_embeddings.py ' \
          '--model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_roberta ' \
          '--pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 128 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          '--num_shards 1 --CSNET_ADV ' \
          '--out_file ' + OUTPUT_ENCODED_FILE #+ ' &>> ' + OUTPUT_DIR + 'log_encoding_'+file_name+ '.txt &'
# print (command)
print(command, flush=True)
os.system(command)








