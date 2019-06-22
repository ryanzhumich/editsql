#! /bin/bash

# Get preprocessed ATIS data from https://github.com/lil-lab/atis. 
# It requires LDC93S5, LDC94S19, and LDC95S26.

LOGDIR="logs_atis"

CUDA_VISIBLE_DEVICES=1 python3 run.py --raw_train_filename="data/atis_data/data/resplit/processed/train_with_tables.pkl" \
               --raw_dev_filename="data/atis_data/data/resplit/processed/dev_with_tables.pkl" \
               --raw_validation_filename="data/atis_data/data/resplit/processed/valid_with_tables.pkl" \
               --raw_test_filename="data/atis_data/data/resplit/processed/test_with_tables.pkl" \
               --data_directory=processed_data_atis \
               --input_key="utterance" \
               --anonymize=1 \
               --anonymization_scoring=1 \
               --use_snippets=1 \
               --state_positional_embeddings=1 \
               --snippet_age_embedding=1 \
               --discourse_level_lstm=1 \
               --interaction_level=1 \
               --reweight_batch=1 \
               --train=1 \
               --logdir=$LOGDIR \
               --evaluate=1 \
               --evaluate_split="dev+test" \
               --compute_metrics=1 \
               --enable_testing=1 \
               --use_predicted_queries=1

python3 eval_scripts/metric_averages.py $LOGDIR/dev_use_predicted_queries_predictions.json > $LOGDIR/dev_eval.txt
python3 eval_scripts/metric_averages.py $LOGDIR/test_use_predicted_queries_predictions.json > $LOGDIR/test_eval.txt
