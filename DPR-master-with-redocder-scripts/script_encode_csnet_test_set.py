
import os

lang='python'
# lang='java'

if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/python/dpr_biencoder.9.7870"
if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code_with_hard_neg/python/dpr_biencoder.3.41969"
# if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_csnet_text_code/java/dpr_biencoder.0.28404" #for java NLP11
if lang=='java': CHECKPOINT="//home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/java/dpr_biencoder.8.5154" #for java NLP11
if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code_with_hard_neg/java/dpr_biencoder.2.27482"




GITHUB_DB="/home/rizwan/CodeBERT/data/codesearch/" #local in NLP10

OUTPUT_DIR ='/local/rizwan/DPR_models/biencoder_models_csnet_text_code_with_hard_neg/'+lang+"/"

pretrained_model="microsoft/graphcodebert-base"


FILE = "/home/rizwan/CodeBERT/data/codesearch/dataset/"+lang+"/codebase.jsonl"

# OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'codebert_encoddings_codebase.jsonl'
OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'with_hard_neg_graphcodebert_encoddings_codebase.jsonl'

DEVICES = [5]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])
mport = 3234


command = "CUDA_VISIBLE_DEVICES="+CUDA_VISIBLE_DEVICES \
     +" python -m torch.distributed.launch --nproc_per_node="+str(len(DEVICES)) \
      + ' --master_port=' + str(mport) + \
          ' generate_dense_embeddings.py ' \
          '--model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_roberta ' \
          '--pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 512 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          '--num_shards 1 --dataset csnet_candidates ' \
          '--out_file ' + OUTPUT_ENCODED_FILE #+ ' &>> ' + OUTPUT_DIR + 'log_encoding_'+file_name+ '.txt &'
# print (command)
print(command, flush=True)
os.system(command)








