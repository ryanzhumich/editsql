#! /bin/bash

# 1. preprocess dataset by the following. It will produce data/sparc_data/

python3 preprocess_sparc.py

# 2. train and evaluate.
#    the result (models, logs, prediction outputs) are saved in $LOGDIR

GLOVE_PATH="/home/lily/rz268/dialog2sql/word_emb/glove.840B.300d.txt" # you need to change this
LOGDIR="logs_sparc_cdseq2seq"

CUDA_VISIBLE_DEVICES=0 python3 run.py --raw_train_filename="data/sparc_data/train.pkl" \
          --raw_validation_filename="data/sparc_data/dev.pkl" \
          --database_schema_filename="data/sparc_data/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="processed_data_sparc" \
          --input_key="utterance" \
          --state_positional_embeddings=1 \
          --discourse_level_lstm=1 \
          --interaction_level=1 \
          --reweight_batch=1 \
          --freeze=1 \
          --train=1 \
          --logdir=$LOGDIR \
          --evaluate=1 \
          --evaluate_split="valid" \
          --use_predicted_queries=1

# 3. get evaluation result

python3 postprocess_eval_sparc.py --pred_file $LOGDIR/valid_use_predicted_queries_predictions.json
