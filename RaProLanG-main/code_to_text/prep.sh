#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
# <path to data> which contains binarized data for each directions
SRCDIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

top_k=${1:-100}
LANG=${2:-c}
WITH_WITHOUT_SUFFIX=${4:-no} # with or no

SPMDIR=/local/wasiahmad/codebart
RETDIR=/local/rizwan/DPR_models/iclr_dpr_code_text_output
SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/iclr_code_text/${WITH_WITHOUT_SUFFIX}_ref_top_${top_k}"
mkdir -p $SAVE_DIR


function spm_preprocess () {

LANG=$1
for SPLIT in test train valid; do
    if [[ ! -f $RETDIR/$LANG/${SPLIT}.spm.$LANG ]]; then
        if [[ $SPLIT == *"test" ]]; then
            MAX_TGT_LEN=9999 # we do not truncate test sequences
            SUFF="overall_"$SPLIT
        else
            MAX_TGT_LEN=128
            SUFF="in_domain_"$SPLIT
        fi
        SRC_FIELD="question"
        python encode.py \
            --top_k $top_k \
            --WITH_WITHOUT_SUFFIX $WITH_WITHOUT_SUFFIX \
            --model-file ${SPMDIR}/sentencepiece.bpe.model \
            --input_file $RETDIR/iclr_code_text_retrieval_iclr_only_dedup_${SUFF}_100.json \
            --output_dir $SAVE_DIR \
            --src_field $SRC_FIELD \
            --tgt_field answers \
            --src_lang $LANG \
            --tgt_lang en_XX \
            --pref $SPLIT \
            --max_src_len 256 \
            --max_tgt_len $MAX_TGT_LEN \
            --workers 60;
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

spm_preprocess 'c'
binarize 'c'