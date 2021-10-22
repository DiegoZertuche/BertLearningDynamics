#!/bin/bash

# Master script to re-generate all edge probing datasets.
# This script just follows the instructions in data/README.md;
# see that file and the individual dataset scripts for more details.
#
# Usage:
#  # First, modify the paths below to point to OntoNotes and SPR1 on your system.
#  ./get_and_process_all_data.sh /path/to/glue_data
#
# Note that OntoNotes is rather large and we need to process it several times, so
# this script can take a while to run.

#JIANT_DATA_DIR=${1:-"$HOME/glue_data"}  # path to glue_data directory

## Configure these for your environment ##
#PATH_TO_ONTONOTES="/nfs/jsalt/home/iftenney/ontonotes/ontonotes/conll-formatted-ontonotes-5.0"
#PATH_TO_SPR1_RUDINGER="/nfs/jsalt/home/iftenney/decomp.net/spr1"

## Don't modify below this line. ##

set -eux


TARGET_DIR=$1
OUTPUT_DIR="${TARGET_DIR}/edges"
TOKENIZER=$2
HERE=$(dirname $0)

function preproc_task() {
    TASK_DIR=$1
    # Extract data labels.
    python $HERE/get_edge_data_labels.py -o $TASK_DIR/labels.txt \
      -i $TASK_DIR/*.json -s

    # Retokenize for each tokenizer we need.
    python $HERE/retokenize_edge_data.py -t "${TOKENIZER}"  $TASK_DIR/*.json

    # Convert the original version to tfrecord.
    python $HERE/convert_edge_data_to_tfrecord.py $TASK_DIR/*.json
}

function get_semeval() {
    ## SemEval 2010 Task 8 relation classification
    ## Gives semeval/{split}.json, where split = {train.0.85, dev, test}
    #mkdir $OUTPUT_DIR/semeval
    #bash $HERE/data/get_semeval_data.sh $OUTPUT_DIR/semeval
    preproc_task $OUTPUT_DIR/semeval
}

get_semeval

