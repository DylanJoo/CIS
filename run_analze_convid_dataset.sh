python3 tools/check_queries_overlapping.py \
  --topic data/canard/train.queries.jsonl \
  --run_target convir_data_construction/runs/cast20.canard.train.answer.spr.top1000.trec \
  --run_reference convir_data_construction/runs/cast20.canard.train.rewrite.spr.top1000.trec \
  --convir_dataset_stats convir_data_construction/cast20.canard.train.triplet.spr.overlapped.stats \
  -k_pos 3 \
  -k 200 \
  -collections data/cast20/collections/
        
