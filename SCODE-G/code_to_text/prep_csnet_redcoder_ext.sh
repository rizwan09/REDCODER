#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
# <path to data> which contains binarized data for each directions
SRCDIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

LANG=${1:-python}
top_k=${2:-5}
WITH_OR_WITHOUT_REF=${3:-with} #no or with are saved and used processed.py (not this file) so they are same here,
RETDIR=${4:-../redcoder_data/retriever_output/codexglue_csnet_code_to_text/}
SAVE_DIR=${5:-../redcoder_data/codexglue_csnet_code_to_text_scode-g-preprocessed-input/}

SPMDIR=../../sentencepiece
mkdir -p $SAVE_DIR


function spm_preprocess () {

LANG=$1
for SPLIT in test train valid; do
    if [[ ! -f $RETDIR/$LANG/${SPLIT}.spm.$LANG ]]; then
        if [[ $SPLIT == "test" ]]; then
            MAX_TGT_LEN=9999 # we do not truncate test sequences
        else
            MAX_TGT_LEN=128

        fi
        SRC_FIELD="question"
        python encode_with_codes.py \
            --top_k $top_k \
            --WITH_WITHOUT_SUFFIX $WITH_WITHOUT_SUFFIX \
            --model-file ${SPMDIR}/sentencepiece.bpe.model \
            --input_file $RETDIR/${LANG}_csnet_code_text_retrieval_dedup_${SPLIT}_100.json \
            --output_dir $SAVE_DIR \
            --src_field $SRC_FIELD \
            --tgt_field answers \
            --src_lang $LANG \
            --tgt_lang en_XX \
            --pref $SPLIT \
            --max_src_len 256 \
            --max_tgt_len $MAX_TGT_LEN \
            --workers 60;
#
    fi
done

}

function binarize () {

LANG=$1
if [[ ! -d $SRCDIR/$LANG/data-bin ]]; then
    fairseq-preprocess \
        --source-lang $LANG \
        --target-lang en_XX \
        --trainpref $SAVE_DIR/train.spm \
        --validpref $SAVE_DIR/valid.spm \
        --testpref $SAVE_DIR/test.spm \
        --destdir $SAVE_DIR/data-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --workers 60 \
        --srcdict ${SPMDIR}/dict.txt \
        --tgtdict ${SPMDIR}/dict.txt;
fi

}

spm_preprocess $LANG
binarize $LANG