# train
python3 tools/construct_convir_dataset.py \
  --query data/canard/train.history.tsv \
  --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
  --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
  --convir_dataset convir_data/canard_convir.train.triples.cqe.v0.jsonl \
  --triplet \
  -k 200 \
  -collections data/rel_collections/CANARD_V1_REL.jsonl

# dev
# python3 tools/construct_convir_dataset.py \
#   --query data/canard/train.history.tsv \
#   --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
#   --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
#   --convir_dataset convir_data/canard_convir.train.triples.cqe.v0.jsonl \
#   --triplet \
#   -k 200 \
#   -collections data/rel_collections/CANARD_V1_REL.jsonl
