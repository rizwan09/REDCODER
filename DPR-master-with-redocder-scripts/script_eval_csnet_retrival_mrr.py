import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]

lang='python'
lang='java'

if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/python/dpr_biencoder.9.7870"
if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code_with_hard_neg/python/dpr_biencoder.3.41969"
# if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_csnet_text_code/java/dpr_biencoder.0.28404" #for java NLP11
if lang=='java': CHECKPOINT="//home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/java/dpr_biencoder.8.5154" #for java NLP11
if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code_with_hard_neg/java/dpr_biencoder.2.27482"


pretrained_model="/home/rizwan/graphcodebert-base"

OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code_with_hard_neg/"+lang



# OUTPUT_ENCODED_FILES = '/local/rizwan/DPR_models/biencoder_models_csnet_text_code/'+lang+'/codebert_encoddings_codebase.jsonl_0.pkl'
OUTPUT_ENCODED_FILES = '/local/rizwan/DPR_models/biencoder_models_csnet_text_code_with_hard_neg/'+lang+'/with_hard_neg_graphcodebert_encoddings_codebase.jsonl_0.pkl'



qa_file = "/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/test.jsonl"

print(qa_file)

DEVICES = [4]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

top_k =  1001

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python csnet_retriever.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ qa_file + \
          '  --qa_file '+ qa_file + \
          '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILES + \
          '  --out_file  '+ OUTPUT_DIR_PATH+"_csnet_test_"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --sequence_length 256 --save_or_load_index '
# print (command)
print(command, flush=True)
os.system(command)




