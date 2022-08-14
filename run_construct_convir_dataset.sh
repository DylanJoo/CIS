# ## V0
# python3 tools/construct_convir_dataset.py \
#   --topic data/canard/train.queries.jsonl \
#   --run_target convir_data_construction/runs/cast20.canard.train.teacher.top1000.monot5.trec \
#   --run_reference convir_data_construction/runs/cast20.canard.train.student.top1000.monot5.trec \
#   --convir_dataset data/canard4ir/cast20.canard.train.triplet.top3.wndw.3.tsv \
#   --version 'top3' \
#   -k_pos 3 \
#   -n 30 \
#   -k 200 \
#   --window_size 3 \
#   -collections data/cast20/collections/
#
#
# ## V1
# python3 tools/construct_convir_dataset.py \
#   --topic data/canard/train.queries.jsonl \
#   --run_target convir_data_construction/runs/cast20.canard.train.teacher.top1000.monot5.trec \
#   --run_reference convir_data_construction/runs/cast20.canard.train.student.top1000.monot5.trec \
#   --convir_dataset data/canard4ir/cast20.canard.train.triplet.overlapped.wndw.3.tsv \
#   --version 'overlapped' \
#   -k_pos 3 \
#   -n 30 \
#   -k 200 \
#   --window_size 3 \
#   -collections data/cast20/collections/

# ## V0
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.queries.jsonl \
  --run_target convir_data_construction/runs/cast20.canard.train.teacher.top1000.monot5.trec \
  --run_reference convir_data_construction/runs/cast20.canard.train.student.top1000.monot5.trec \
  --convir_dataset data/canard4ir/cast20.canard.train.triplet.top3.wndw.allq.tsv \
  --version 'top3' \
  -k_pos 3 \
  -n 30 \
  -k 200 \
  --discard_history_responses \
  -collections data/cast20/collections/


## V1
python3 tools/construct_convir_dataset.py \
  --topic data/canard/train.queries.jsonl \
  --run_target convir_data_construction/runs/cast20.canard.train.teacher.top1000.monot5.trec \
  --run_reference convir_data_construction/runs/cast20.canard.train.student.top1000.monot5.trec \
  --convir_dataset data/canard4ir/cast20.canard.train.triplet.overlapped.wndw.allq.tsv \
  --version 'overlapped' \
  -k_pos 3 \
  -n 30 \
  -k 200 \
  --discard_history_responses \
  -collections data/cast20/collections/
