#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/pre-trained_language_models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"


echo "lowercase models"

echo "BERT BASE LOWERCASED"
if [[ ! -f bert/uncased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
  unzip uncased_L-12_H-768_A-12.zip
  rm uncased_L-12_H-768_A-12.zip
  cd uncased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
  tar -xzf bert-base-uncased.tar.gz
  rm bert-base-uncased.tar.gz
  rm bert_model*
  cd ../../
fi


echo 'cased models'

echo "BERT BASE CASED"
if [[ ! -f bert/cased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
  unzip cased_L-12_H-768_A-12
  rm cased_L-12_H-768_A-12.zip
  cd cased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz"
  tar -xzf bert-base-cased.tar.gz
  rm bert-base-cased.tar.gz
  rm bert_model*
  cd ../../
fi

cd "$ROOD_DIR"
echo 'Building common vocab'
if [ ! -f "$DST_DIR/common_vocab_lowercased.txt" ]; then
  python lama/vocab_intersection.py
else
  echo 'Already exists. Run to re-build:'
  echo 'python util_KB_completion.py'
fi

