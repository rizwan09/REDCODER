
import os

lang='python'
# lang='java'


# if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_csnet_text_code/python/dpr_biencoder.2.17175" #for python NLP10
if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/python/dpr_biencoder.9.7870"
# if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_csnet_text_code/java/dpr_biencoder.0.28404" #for java NLP11
if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/java/dpr_biencoder.8.5154" #for java NLP11


GITHUB_DB="/local/rizwan/workspace/projects/RaProLanG/data/plbart/" #local in NLP10

OUTPUT_DIR ='/local/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code/'+lang+"/"

pretrained_model="microsoft/graphcodebert-base"
pretrained_model="/home/rizwan/graphcodebert-base/"


FILE_EXT=lang+'_dedupe_definitions_v2.pkl'
FILE =  GITHUB_DB+lang+"/"+FILE_EXT
OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'encoddings_'  +FILE_EXT

DEVICES = [3,4,5]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES + \
          ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
          ' generate_dense_embeddings.py ' \
          '--model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_roberta ' \
          '--pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 512 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          '--num_shards 1 ' \
          '--out_file ' + OUTPUT_ENCODED_FILE #+ ' &>> ' + OUTPUT_DIR + 'log_encoding_'+file_name+ '.txt &'
# print (command)
print(command, flush=True)
os.system(command)








