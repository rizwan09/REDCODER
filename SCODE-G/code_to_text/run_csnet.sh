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







SPM_MODEL=CODE_DIR_HOME/sentencepiece/sentencepiece.bpe.model

GPU=${1:-2}
SOURCE=${2:-java}
PATH_2_DATA=${3:-../redcoder_data/codexglue_csnet_code_to_text_scode-g-preprocessed-input/}
PRETRAIN=${4:-./checkpoint_11_100000.pt}
SAVE_DIR=${5:-../redcoder_data/codexglue_csnet_code_to_text_scode-g-output/}
UPDATE_FREQ=${6:-8}
BATCH_SIZE=${7:-16}

TARGET=en_XX

mkdir -p $SAVE_DIR

export CUDA_VISIBLE_DEVICES=$GPU


echo "Source: $SOURCE Target: $TARGET"


USER_DIR=""
TASK=translation_from_pretrained_bart



function fine_tune () {
	OUTPUT_FILE=${SAVE_DIR}/finetune.log
	fairseq-train $PATH_2_DATA $USER_DIR \
		--restore-file $PRETRAIN \
		--bpe 'sentencepiece' --sentencepiece-model $SPM_MODEL \
  		--langs $langs --arch mbart_base --layernorm-embedding \
  		--task $TASK --source-lang $SOURCE --target-lang $TARGET \
  		--criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  		--batch-size ${BATCH_SIZE} --update-freq ${UPDATE_FREQ} --max-epoch 15 \
  		--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  		--lr-scheduler polynomial_decay --lr 5e-05 --min-lr -1 \
  		--warmup-updates 1000 --max-update 200000 \
  		--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  		--seed 1234 --log-format json --log-interval 100 \
  		--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
		--eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
  		--eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
  		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  		--eval-bleu-print-samples --no-epoch-checkpoints --patience 5 \
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

	gold_ref=$FILE_PREF.ref #ideal practice is: this should be ${PATH_2_DATA}/test.target but this should be similar in performance I followed from previous work and did not change.

	printf "CodeXGlue Blue-4 Evaluation: \t" >> ${RESULT_FILE}
	python evaluator.py $gold_ref $FILE_PREF.hyp >> ${RESULT_FILE}
	python evaluator.py $gold_ref $FILE_PREF.hyp;

    printf "CodeXGlue Rouge-l Evaluation: \t" >> ${RESULT_FILE}
	files2rouge  $gold_ref $FILE_PREF.hyp >> ${RESULT_FILE}
	files2rouge  $gold_ref $FILE_PREF.hyp;
}


fine_tune
generate

