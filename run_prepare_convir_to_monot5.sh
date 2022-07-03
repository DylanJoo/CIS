python3 tools/convert_run_to_monot5.py \
  -run spr/runs/train.canard.answer.top1000.trec \
  -corpus data/rel_collections/CANARD_V1_REL.jsonl \
  -k 1000 \
  -q data/canard/train.rewrite.tsv \
  --output_text_pair train.canard.answer.top1000.text_pair.txt \
  --output_id_pair train.canard.answer.top1000.id_pair.txt

