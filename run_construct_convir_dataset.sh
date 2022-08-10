## V0
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.queries.jsonl \
  --run_target convir_data_construction/runs/cast20.canard.train.teacher.top1000.monot5.trec \
  --run_reference convir_data_construction/runs/cast20.canard.train.student.top1000.monot5.trec \
  --convir_dataset data/canard4ir/cast20.canard.train.triplet.top3.tsv \
  --version 'top3' \
  -k_pos 3 \
  -k 200 \
  -collections data/cast20/collections/
        

## V1
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.queries.jsonl \
  --run_target convir_data_construction/runs/cast20.canard.train.teacher.top1000.monot5.trec \
  --run_reference convir_data_construction/runs/cast20.canard.train.student.top1000.monot5.trec \
  --convir_dataset data/canard4ir/cast20.canard.train.triplet.overlapped.tsv \
  --version 'overlapped' \
  -k_pos 3 \
  -k 200 \
  -collections data/cast20/collections/
