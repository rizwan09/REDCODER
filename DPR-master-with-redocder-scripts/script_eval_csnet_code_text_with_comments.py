import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]

lang='python'
lang='java'

if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text/python/dpr_biencoder.14.10493"
if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text/java/dpr_biencoder.14.6872"


pretrained_model="/home/rizwan/graphcodebert-base"


OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text_with_comments/"+lang
OUTPUT_DIR_PATH="/local/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text_ret_only_from_csent"+lang



# OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text/"+lang+'/encoddings_deduplicated.summaries.txt_0.pkl'
OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text_ret_only_from_csent/"+lang+"/encoddings_"+lang+"_dedupe_definitions_v2.pkl_0.pkl"

split=["train", "valid", "test"][1]

if split=="test": qa_file = "/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/test.jsonl"
if split=="train": qa_file = "/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/train.jsonl"
if split=="valid": qa_file ="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/valid.jsonl"



CTX_FILE =  FILE = './deduplicated.summaries.txt'

GITHUB_DB="/local/rizwan/workspace/projects/RaProLanG/data/plbart/"
FILE_EXT=lang+'_dedupe_definitions_v2.pkl'
CTX_FILE =  FILE =  GITHUB_DB+lang+"/"+FILE_EXT



print(qa_file)

DEVICES = [5]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])


top_k =  50

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python dense_retriever_with_comments.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ CTX_FILE + \
          '  --qa_file '+ qa_file + \
          '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILES + \
          '  --out_file  '+ OUTPUT_DIR_PATH+"_csnet_code_text_retrieval_dedup_"+split+"_"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --sequence_length 256 --save_or_load_index --code_to_text'
# print (command)
print(command, flush=True)
os.system(command)



