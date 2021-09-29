#!/usr/bin/env bash
# run: bash prepare.sh java 1 false false true &

LANG=${1:-python}
top_k=${2:-5}
WITH_OR_WITHOUT_REF=${3:-with} #no or with are saved and used processed.py (not this file) so they are same here,
# use without for not ref which is no in other cases

SPMDIR=/local/wasiahmad/codebart
RETDIR=/local/rizwan/DPR_models/csnet_with_comments
SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet_with_comments/${LANG}_${WITH_OR_WITHOUT_REF}_ref_top_${top_k}"
mkdir -p $SAVE_DIR


function prepare () {


if [[ "$NO_CONCODE_VARS" = true ]]; then
    EXTRA=" --NO_CONCODE_VARS "
else
    EXTRA=""
fi



for split in train valid test; do
    python process_with_comments.py \
        --retrieved_code_file $RETDIR/${LANG}_csnet_pos_only_retrieval_dedup_${split}_30.json \
        --split $split \
        --top_k $top_k \
        --out_dir $SAVE_DIR \
        --WITH_OR_WITHOUT_REF $WITH_OR_WITHOUT_REF \
        $EXTRA;
done

}

function spm_preprocess () {

LANG=$1
for SPLIT in train valid test; do
    if [[ ! -f $SAVE_DIR/${SPLIT}.spm.$LANG ]]; then
        python encode.py \
            --model_file $SPMDIR/sentencepiece.bpe.model \
            --input_source $SAVE_DIR/$SPLIT.source \
            --input_target $SAVE_DIR/$SPLIT.target \
            --output_source $SAVE_DIR/${SPLIT}.spm.en_XX \
            --output_target $SAVE_DIR/${SPLIT}.spm.$LANG \
            --max_len 510 \
            --workers 60;
    fi
done

}

function binarize () {

LANG=$1
if [[ ! -d $SAVE_DIR/data-bin ]]; then
    fairseq-preprocess \
        --source-lang en_XX \
        --target-lang $LANG \
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

for lang in $LANG; do
    prepare
    spm_preprocess $lang
    binarize $lang
done