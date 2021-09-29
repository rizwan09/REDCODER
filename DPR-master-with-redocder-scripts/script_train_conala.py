import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]
lang = "python"
# lang = "java"
# lang = "javascript"
# lang = "go"

text_to_code=True

CHECKPOINT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_conala/"

pretrained_model="microsoft/codebert-base"
pretrained_model="/home/rizwan/graphcodebert-base/"

DEVICES=[0,1]
CUDA_VISIBLE_DEVICES=','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES="+CUDA_VISIBLE_DEVICES+" python  -m torch.distributed.launch --nproc_per_node="+str(len(DEVICES))+" train_dense_encoder.py \
 --max_grad_norm 2.0 \
 --encoder_model_type hf_roberta \
 --pretrained_model_cfg " + pretrained_model   + " \
 --eval_per_epoch 1 \
 --seed 12345 \
 --sequence_length 256 \
 --warmup_steps 1237 \
 --batch_size 8 \
 --train_file /home/rizwan/DPR_models/conala-corpus/conala-combined-train.json   \
 --dev_file /home/rizwan/DPR_models/conala-corpus/conala-combined-train.json \
 --output_dir "+CHECKPOINT_DIR_PATH+" \
 --learning_rate 2e-5 \
 --num_train_epochs 30 \
 --dev_batch_size 64 \
 --val_av_rank_start_epoch 0 \
 --fp16 --dataset conala\
"
if text_to_code: command+= ' --text_to_code '
print (command, flush=True)
os.system(command)












