#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
# <path to data> which contains binarized data for each directions
SRCDIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

langs=java,python,en_XX

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID SRC_LANG"
            echo
            echo "SRC_LANG  Language choices: [java, python, go, javascript, php, ruby, c]"
            echo
            exit;;
    esac
done



SOURCE=${1:-c}
TARGET=en_XX
top_k=${2:-100}
WITH_WITHOUT_SUFFIX=${3:-no}
GPU=${4:-0}

export CUDA_VISIBLE_DEVICES=$GPU

PRETRAINED_CP_NAME=checkpoint_11_100000.pt

PRETRAIN=/local/wasiahmad/codebart/checkpoints/${PRETRAINED_CP_NAME}
PATH_2_DATA=/local/rizwan/workspace/projects/RaProLanG/data/plbart/iclr_code_text/${WITH_WITHOUT_SUFFIX}_ref_top_${top_k}/data-bin
SPM_MODEL=/local/wasiahmad/codebart/sentencepiece.bpe.model


echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=/local/rizwan/workspace/projects/RaProLanG/iclr_code_text/${SOURCE}_${TARGET}_${WITH_WITHOUT_SUFFIX}_${top_k}
mkdir -p ${SAVE_DIR}

if [[ "$SOURCE" =~ ^(ruby|javascript|go|php|c)$ ]]; then
    USER_DIR="--user-dir "${CODE_DIR_HOME}/FID
    TASK=translation_in_same_language
else
    USER_DIR=""
    TASK=translation_from_pretrained_bart
fi


function fine_tune () {
	OUTPUT_FILE=${SAVE_DIR}/finetune.log
	fairseq-train $PATH_2_DATA $USER_DIR \
		--restore-file $PRETRAIN \
		--bpe 'sentencepiece' --sentencepiece-model $SPM_MODEL \
  		--langs $langs --arch mbart_base --layernorm-embedding \
  		--task $TASK --source-lang $SOURCE --target-lang $TARGET \
  		--criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  		--batch-size 32 --update-freq 4 --max-epoch 15 \
  		--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  		--lr-scheduler polynomial_decay --lr 5e-05 --min-lr -1 \
  		--warmup-updates 1000 --max-update 200000 \
  		--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  		--seed 1234 --log-format json --log-interval 100 \
  		--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
		--eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
  		--eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
  		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  		--eval-bleu-print-samples --no-epoch-checkpoints --patience 5 --fp16 \
  		--ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE}
}


function generate () {
	model=${SAVE_DIR}/checkpoint_best.pt
	FILE_PREF=${SAVE_DIR}/output
	RESULT_FILE=${SAVE_DIR}/result.txt

	fairseq-generate $PATH_2_DATA $USER_DIR \
 		--path $model \
  		--task $TASK \
  		--gen-subset test \
  		-t $TARGET -s $SOURCE \
  		--sacrebleu --remove-bpe 'sentencepiece' \
  		--batch-size 4 --langs $langs --beam 10 > $FILE_PREF

	cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp
	cat $FILE_PREF | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.ref
	sacrebleu -tok 'none' -s 'none' $FILE_PREF.ref < $FILE_PREF.hyp 2>&1 | tee ${RESULT_FILE}
	printf "CodeXGlue Blue-4 Evaluation: \t" >> ${RESULT_FILE}
	python evaluator.py $FILE_PREF.ref $FILE_PREF.hyp >> ${RESULT_FILE}
	python evaluator.py $FILE_PREF.ref $FILE_PREF.hyp;

    printf "CodeXGlue Rouge-l Evaluation: \t" >> ${RESULT_FILE}
	files2rouge  $FILE_PREF.ref $FILE_PREF.hyp >> ${RESULT_FILE}
	files2rouge  $FILE_PREF.ref $FILE_PREF.hyp;
}


fine_tune
generate
