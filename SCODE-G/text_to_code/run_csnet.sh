


#!/usr/bin/env bash

#bash run.sh 0,1 java false -1

export PYTHONIOENCODING=utf-8;
# <path to data> which contains binarized data for each directions
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

USER_NAME=`whoami`

SPM_MODEL=CODE_DIR_HOME/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX #,go,php,ruby,javascript

task=translation_from_pretrained_bart

GPU=${1:-2}
LANG=${2:-"python"}
path_2_data=${3:-../redcoder_data/codexglue_csnet_text_to_code_scode-g-preprocessed-input/}
PRETRAIN=${4:-./checkpoint_11_100000.pt}
SAVE_DIR=${5:-../redcoder_data/codexglue_csnet_text_to_code_scode-g-output/}
UPDATE_FREQ=${6:-8}
BATCH_SIZE=${7:-4}
USE_PLBART=${8:-true}

mkdir -p $SAVE_DIR

evaluator_script="${CODE_DIR_HOME}/evaluation_scripts/evaluator.py";
codebleu_path="${CODE_DIR_HOME}/evaluation_scripts/CodeBLEU";

export CUDA_VISIBLE_DEVICES=$GPU



# read the split words into an array based on comma delimiter
IFS=',' read -a GPU_IDS <<< "$GPU"
NUM_GPUS=${#GPU_IDS[@]}
BSZ=$((BATCH_SIZE*UPDATE_FREQ*NUM_GPUS))

# CSNET data size is as follows
# java: 165k, python: 252k, php: 241k, go: 167k, js: 58k, ruby:25k
# So, number of mini-batches for each language would be:
# java: ~5100, python: ~7800, php: ~7500, go: ~5200, js: ~1800, ruby: ~780

MAX_UPDATE=100000

declare -A WARMUP
WARMUP['java']=5000
WARMUP['python']=5000
WARMUP['php']=5000
WARMUP['go']=5000
WARMUP['js']=2000
WARMUP['ruby']=1000

EXTRA_PARAM=""
if [[ "$USE_PLBART" = true ]]; then
    EXTRA_PARAM="--restore-file $PRETRAIN --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler"
else
    echo "finetuning from scratch without PLBART"
fi




function train () {

fairseq-train "$path_2_data/data-bin" \
    $EXTRA_PARAM \
    --bpe 'sentencepiece' \
    --sentencepiece-model $SPM_MODEL \
    --arch mbart_base \
    --layernorm-embedding \
    --langs $langs \
    --task $task \
    --source-lang en_XX \
    --target-lang $LANG \
    --batch-size $BATCH_SIZE \
    --update-freq $UPDATE_FREQ \
    --fp16 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.2 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --lr 3e-05 \
    --warmup-updates ${WARMUP[${LANG}]} \
    --max-update $MAX_UPDATE \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.1 \
    --eval-bleu \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 5}' \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --patience 5 \
    --seed 1234 \
    --log-format json \
    --log-interval 100 \
    --save-dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train.log;

}

function evaluate () {

FILE_PREF=${SAVE_DIR}/output;
RESULT_FILE=${SAVE_DIR}/result.txt;
GOUND_TRUTH_PATH="${path_2_data}/test.target";
model_path=$SAVE_DIR

fairseq-generate "${path_2_data}/data-bin" \
    --fp16 \
    --path ${model_path}/checkpoint_best.pt \
    --task $task \
    --langs $langs \
    --gen-subset test \
    --source-lang en_XX \
    --target-lang $LANG \
    --sacrebleu \
    --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --beam 10 > ${FILE_PREF};


cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- > $FILE_PREF.hyp;
python $evaluator_script --references ${GOUND_TRUTH_PATH} --predictions ${FILE_PREF}.hyp 2>&1 | tee ${RESULT_FILE};
cd $codebleu_path;
python calc_code_bleu.py --refs ${GOUND_TRUTH_PATH} --hyp $FILE_PREF.hyp --lang $LANG 2>&1 | tee -a ${RESULT_FILE};
cd ${CURRENT_DIR};

}


train
evaluate










