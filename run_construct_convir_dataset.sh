# train
## cqe (v0) 
# python3 tools/construct_convir_dataset.py \
#   --query data/canard/train.history.tsv \
#   --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
#   --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
#   --convir_dataset convir_data/canard_convir.train.triples.cqe.v0.jsonl \
#   --triplet \
#   -k_pos 3 \
#   -k 200 \
#   -collections data/rel_collections/CANARD_V1_REL.jsonl

## cqe (v1) 
# python3 tools/construct_convir_dataset.py \
#   --query data/canard/train.history.tsv \
#   --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
#   --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
#   --convir_dataset convir_data/canard_convir.train.triples.cqe.v1.jsonl \
#   --triplet \
#   --version 'v1' \
#   -k_pos 3 \
#   -k 200 \
#   -collections data/rel_collections/CANARD_V1_REL.jsonl

## licqe (v0) 
python3 tools/construct_convir_dataset.py \
  --query data/canard/train.history.tsv \
  --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
  --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
  --convir_dataset convir_data/canard_convir.train.quadruples.cqe.v0.jsonl \
  --quadruplet \
  --version 'v0' \
  -k_pos 3 \
  -k 200 \
  -collections data/rel_collections/CANARD_V1_REL.jsonl

## licqe (v1) 
python3 tools/construct_convir_dataset.py \
  --query data/canard/train.history.tsv \
  --run_target re-ranking/runs/train.canard.answer.monot5.pred.top1000.rerank.trec \
  --run_reference re-ranking/runs/train.canard.rewrite.monot5.pred.top1000.rerank.trec \
  --convir_dataset convir_data/canard_convir.train.quadruples.cqe.v1.jsonl \
  --quadruplet \
  --version 'v1' \
  -k_pos 3 \
  -k 200 \
  -collections data/rel_collections/CANARD_V1_REL.jsonl

# dev
# python3 tools/construct_convir_dataset.py \
#   --query data/canard/dev.history.tsv \
#   --run_target re-ranking/runs/dev.canard.answer.monot5.pred.top1000.rerank.trec \
#   --run_reference re-ranking/runs/dev.canard.rewrite.monot5.pred.top1000.rerank.trec \
#   --convir_dataset convir_data/canard_convir.dev.triples.cqe.v0.jsonl \
#   --triplet \
#   -k 200 \
#   -collections data/trec-car+marco-psg/
