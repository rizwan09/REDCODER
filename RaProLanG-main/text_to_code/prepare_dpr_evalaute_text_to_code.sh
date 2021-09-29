#!/usr/bin/env bash
# run: bash prepare.sh java 1 false false true &

LANG=${1:-"java"}
top_k=${2:-5}
NO_CONCODE_VARS=${3:-false}
MASK_RATE=${3:-0}
WITH_WITHOUT_SUFFIX=${4:-with} # with or no

SPMDIR=/local/wasiahmad/codebart
RETDIR=/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/${LANG}_${WITH_WITHOUT_SUFFIX}_ref_top_${top_k}_filtered_comments/
SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/${LANG}_DPR_RET_CODE_1_NO_NL"
mkdir -p $SAVE_DIR

export PYTHONIOENCODING=utf-8;
# <path to data> which contains binarized data for each directions
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

evaluator_script="${CODE_DIR_HOME}/evaluation_scripts/evaluator.py";
codebleu_path="${CODE_DIR_HOME}/evaluation_scripts/CodeBLEU";


function prepare () {

if [[ "$NO_CONCODE_VARS" = true ]]; then
    EXTRA=" --NO_CONCODE_VARS "
else
    EXTRA=""
fi



for split in test; do
    python dpr_process.py \
        --input_data_dir $RETDIR \
        --output_data_dir $SAVE_DIR

done


python $evaluator_script --references ${SAVE_DIR}/test.target --predictions ${SAVE_DIR}/test.source 2>&1 ;
cd $codebleu_path;
python calc_code_bleu.py --refs ${SAVE_DIR}/test.target --hyp ${SAVE_DIR}/test.source  --lang $LANG 2>&1 ;
cd ${CURRENT_DIR};
}

for lang in $LANG; do
    prepare
done


