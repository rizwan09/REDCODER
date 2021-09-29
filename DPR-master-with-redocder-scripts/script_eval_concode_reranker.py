import os

lang='java'


# CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/dpr_biencoder.5.1786"
CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_with_concode_tokens/java/dpr_biencoder.6.2084"

pretrained_model="microsoft/codebert-base"

# OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/"
OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_concode_with_code_tokens/java/"
BASE_COMMENT_DIR='/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/'
# OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/github_encoddings_conode/java/concode_train.json_0.pkl"
OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/github_encoddings_conode/java/concode_with_code_token_train.json_"
# ctx_file="/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/train.json"
ctx_file="/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/"


dataset='CONCODE'


pretrained_model="microsoft/codebert-base"


qa_file_suffix=["train", "valid", "test"][2]
qa_file = BASE_COMMENT_DIR+str(qa_file_suffix)+'.json'
print(qa_file)

DEVICES = [2]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

top_k =  1
ctx_file_1='/home/rizwan/DPR_models/retrieved.txt'
ctx_file_2='/local/rizwan/workspace/projects/RaProLanG/plbart_ori_top_1_masked/-java-ms100000-wu5000-bsz64/output.hyp'

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python reranker_retriever.py  ' \
          ' --model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_roberta ' \
          ' --pretrained_model_cfg   ' + pretrained_model + \
          '  --qa_file '+ qa_file + \
          '  --out_file  /home/rizwan/DPR_models/pred_reranked.txt ' \
          '  --n-docs  '+ str(top_k) + ' --batch_size 1 --match exact ' \
          ' --sequence_length 256 --save_or_load_index --dataset CONCODE ' \
          ' --ctx_file_1 ' + ctx_file_1 + ' --ctx_file_2 '+ctx_file_2 #+ ' --ctx_file_3 '+ctx_file_3
# print (command)
# \ # ' -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
print(command, flush=True)
os.system(command)






