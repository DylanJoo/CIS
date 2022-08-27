python3 tools/construct_denoise_convir_dataset.py \
  --topic data/canard/train.queries.jsonl \
  --run_teacher convir_data_construction/runs/cast20.canard.train.teacher.top1000.monot5.trec \
  --run_student convir_data_construction/runs/cast20.canard.train.student.top1000.monot5.trec \
  --output data/canard4ir/canard4ir.train.monot5.triplet.denoise.top3.tsv \
  --topk_pool 200 \
  --topk_positive 3 \
  --n 30 \
  -collections data/cast20/collections/

