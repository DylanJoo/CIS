python3 xla_spawn.py \
  --num_cores 8 \
  train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir ./checkpoints/samples \
  --train_file ../convir_data/train.triples.sample.jsonl \
  --eval_file ../convir_data/train.triples.sample.jsonl \
  --max_q_seq_length 32 \
  --max_p_seq_length 128 \
  --colbert_type 'colbert' \
  --dim 128 \
  --remove_unused_columns false \
  --per_device_train_batch_size 8 \
  --evaluation_strategy 'steps'\
  --max_steps 100 \
  --save_steps 100 \
  --eval_steps 50 \
  --do_train \
  --do_eval
