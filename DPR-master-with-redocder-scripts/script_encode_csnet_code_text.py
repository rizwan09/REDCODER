
import os

lang='python'
# lang='java'

ret_only_from_csent=False


if lang=='python': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text/python/dpr_biencoder.14.10493"
if lang=='java': CHECKPOINT="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text/java/dpr_biencoder.14.6872"


OUTPUT_DIR ='/local/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text/'+lang+"/"

pretrained_model="microsoft/graphcodebert-base"
pretrained_model="/home/rizwan/graphcodebert-base/"


FILE = './deduplicated.summaries.txt'

OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'encoddings_deduplicated.summaries.txt'

if ret_only_from_csent:
    GITHUB_DB="/local/rizwan/workspace/projects/RaProLanG/data/plbart/"
    FILE_EXT=lang+'_dedupe_definitions_v2.pkl'
    FILE =  GITHUB_DB+lang+"/"+FILE_EXT
    OUTPUT_DIR = '/local/rizwan/DPR_models/biencoder_graphcode_models_csnet_code_text_ret_only_from_csent/' + lang + "/"
    OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'encoddings_'  +FILE_EXT


DEVICES = [4,5]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])
mport = 4234


command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES + \
          ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) \
          + ' --master_port=' + str(mport) \
            +' generate_dense_embeddings.py ' \
          '--model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_roberta ' \
          '--pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 512 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          '--num_shards 1 --code_to_text ' \
          '--out_file ' + OUTPUT_ENCODED_FILE #+ ' &>> ' + OUTPUT_DIR + 'log_encoding_'+file_name+ '.txt &'
# print (command)
print(command, flush=True)
os.system(command)








