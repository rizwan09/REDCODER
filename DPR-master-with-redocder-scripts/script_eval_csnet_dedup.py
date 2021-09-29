import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]

lang='python'
# lang='java'


# if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_csnet_text_code/python/dpr_biencoder.2.17175" #for python NLP10
if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/python/dpr_biencoder.9.7870"
# if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_csnet_text_code/java/dpr_biencoder.0.28404" #for java NLP11
if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/java/dpr_biencoder.8.5154" #for java NLP11


pretrained_model="/home/rizwan/graphcodebert-base"


OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/"+lang


OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/"+lang+"/encoddings_"+lang+"_dedupe_definitions_v2.pkl_0.pkl"

split=["train", "valid", "test"][1]

if split=="test": qa_file = "/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/test.jsonl"
if split=="train": qa_file = "/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/train.jsonl"
if split=="valid": qa_file ="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/valid.jsonl"




GITHUB_DB="/local/rizwan/workspace/projects/RaProLanG/data/plbart/" #local in NLP10
FILE_EXT=lang+'_dedupe_definitions_v2.pkl'
CTX_FILE =  GITHUB_DB+lang+"/"+FILE_EXT


print(qa_file)

DEVICES = [3]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

top_k =  30

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python dense_retriever.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ CTX_FILE + \
          '  --qa_file '+ qa_file + \
          '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILES + \
          '  --out_file  '+ OUTPUT_DIR_PATH+"_csnet_pos_only_retrieval_dedup_"+split+"_"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --sequence_length 256 --save_or_load_index '
# print (command)
print(command, flush=True)
os.system(command)



