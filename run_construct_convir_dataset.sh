## V0
python3 tools/construct_convir_dataset.py \
  --query data/canard/train.history.tsv \
  --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
  --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
  --convir_dataset convir_data/canard.convir.train.quadruples.top3.jsonl \
  --quadruplet \
  --version 'v0' \
  -k_pos 3 \
  -k 200 \
  -collections data/cast22/collections/
        

## V1
python3 tools/construct_convir_dataset.py \
  --query data/canard/train.history.tsv \
  --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
  --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
  --convir_dataset convir_data/canard_convir.train.quadruples.overlapped.jsonl \
  --quadruplet \
  --version 'v1' \
  -k_pos 3 \
  -k 200 \
  -collections data/cast22/collections/
