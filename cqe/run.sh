python3 train_dev.py \
  --resume_from_checkpoint ./checkpoints/colbertv2.0 \
  --kd_teacher_model_name_or_path ./checkpoints/colbertv2.0 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir ./checkpoints/tctccqe.v0 \
  --train_file ../convir_data/canard_convir.train.convtriples.cqe.v0.jsonl \
  --eval_file ../convir_data/canard_convir.dev.triples.cqe.v0.jsonl \
  --max_q_seq_length 32 \
  --max_p_seq_length 150 \
  --colbert_type 'tctcolbert' \
  --dim 128 \
  --remove_unused_columns false \
  --per_device_train_batch_size 12 \ 
  --evaluation_strategy 'steps'\
  --learning_rate 7e-6 \ 
  --max_steps 15000 \
  --save_steps 5000 \
  --eval_steps 5000 \
  --do_train \
  --do_eval
