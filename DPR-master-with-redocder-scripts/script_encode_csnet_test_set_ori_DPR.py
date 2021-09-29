
import os

lang='python'
lang='java'

pretrained_model="bert-base"

OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code_with_hard_neg/"+lang
CHECKPOINT="/home/rizwan/DPR/DPR_ORIGINAL_CP/hf_bert_base.cp"







GITHUB_DB="/home/rizwan/CodeBERT/data/codesearch/" #local in NLP10

OUTPUT_DIR ='/local/rizwan/DPR_models/biencoder_models_csnet_text_code_with_hard_neg/'+lang+"/"

FILE = "/home/rizwan/CodeBERT/data/codesearch/dataset/"+lang+"/codebase.jsonl"

# OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'codebert_encoddings_codebase.jsonl'
OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'ori_dpr_encoddings_codebase.jsonl'

DEVICES = [4]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])
mport = 2234


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








