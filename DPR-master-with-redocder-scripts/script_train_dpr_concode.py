import os

lang = "java"

text_to_code=True
concode_with_code=True


CHECKPOINT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_with_concode_tokens_graphcodebert/"+lang
if not concode_with_code: CHECKPOINT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_concode_with_code_tokens/"+lang

pretrained_model="/home/rizwan/graphcodebert-base/"


DEVICES=[1]
CUDA_VISIBLE_DEVICES=','.join([str(i) for i in DEVICES])

mport = 2234


command = "CUDA_VISIBLE_DEVICES="+CUDA_VISIBLE_DEVICES \
     +" python -m torch.distributed.launch --nproc_per_node="+str(len(DEVICES)) \
      + ' --master_port=' + str(mport) +" train_dense_encoder.py \
    --max_grad_norm 2.0 \
     --encoder_model_type hf_roberta \
     --pretrained_model_cfg " + pretrained_model   + " \
     --eval_per_epoch 1 \
     --seed 12345 \
     --sequence_length 256 \
     --warmup_steps 1237 \
     --batch_size 8 \
     --train_file /home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/train.json   \
     --dev_file /home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/valid.json \
     --output_dir "+CHECKPOINT_DIR_PATH+" \
     --learning_rate 2e-5 \
     --num_train_epochs 10 \
     --dev_batch_size 16 \
     --val_av_rank_start_epoch 1 \
    --fp16 --dataset CONCODE \
"
if text_to_code: command+= ' --text_to_code '
if concode_with_code: command+= ' --concode_with_code '
print (command, flush=True)
os.system(command)








