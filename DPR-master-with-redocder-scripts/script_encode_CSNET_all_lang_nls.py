
import os

lang='python'
# lang='java'
#
# CHECKPOINT="/home/rizwan/DPR_models/biencoder_models/python/dpr_biencoder.2.6441"
# CHECKPOINT="/home/rizwan/DPR_models/biencoder_models/java/dpr_biencoder.1.7101"
CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_codexglue_text_code/python/dpr_biencoder.3.3935"


GITHUB_DB="/home/rizwan/CodeBERT/data/codesearch/train_valid/"+lang+"/" #local in NLP10

OUTPUT_DIR ='/local/rizwan/DPR_models/biencoder_models_codexglue_text_code/'+lang+"/"

pretrained_model="microsoft/codebert-base"

files = 'test.functions_class.tok ' \
        'test.functions_standalone.tok ' \
        'valid.functions_class.tok  ' \
        'valid.functions_standalone.tok ' \
        'train.0.functions_class.tok ' \
        'train.1.functions_class.tok ' \
        'train.2.functions_class.tok ' \
        'train.3.functions_class.tok ' \
        'train.4.functions_class.tok ' \
        'train.5.functions_class.tok ' \
        'train.6.functions_class.tok ' \
        'train.7.functions_class.tok ' \
        'train.0.functions_standalone.tok ' \
        'train.1.functions_standalone.tok ' \
        'train.2.functions_standalone.tok ' \
        'train.3.functions_standalone.tok ' \
        'train.4.functions_standalone.tok ' \
        'train.5.functions_standalone.tok ' \
        'train.6.functions_standalone.tok ' \
        'train.7.functions_standalone.tok'.split()

# print(files)



# for id in range (1): #(len(files)):
#     FILE =  GITHUB_DB+files[-(id+1)]
file_name='train.7.functions_standalone.tok'
FILE =  GITHUB_DB+file_name
OUTPUT_ENCODED_FILE = OUTPUT_DIR + 'less_noisy_codexglue_github_encoddings_'  +file_name

DEVICES = [0,1,2,5,3,]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES + \
          ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
          ' generate_dense_embeddings.py ' \
          '--model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_roberta ' \
          '--pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 128 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          '--num_shards 1 ' \
          '--out_file ' + OUTPUT_ENCODED_FILE #+ ' &>> ' + OUTPUT_DIR + 'log_encoding_'+file_name+ '.txt &'
# print (command)
print(command, flush=True)
os.system(command)








