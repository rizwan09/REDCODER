
import os

lang='c'
CHECKPOINT='/home/rizwan/DPR_models/biencoder_graphcode_models_iclr_code_text/dpr_biencoder.14.5270'


OUTPUT_DIR ='/local/rizwan/DPR_models/biencoder_graphcode_models_iclr_code_text/'

pretrained_model="microsoft/graphcodebert-base"
pretrained_model="/home/rizwan/graphcodebert-base/"


FILE = './iclr_only.deduplicated.summaries.txt'
OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'encoddings_iclr_only_deduplicated.summaries.txt'

DEVICES = [0]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])
mport = 3234


command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES  \
          + ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) \
          + ' --master_port=' + str(mport) \
          + ' generate_dense_embeddings.py ' \
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








