python3 tools/convert_run_to_monot5.py \
  -run spr/runs/dev.canard.rewrite.top1000.trec \
  -corpus data/trec-car+marco-psg \
  -k 1000 \
  -q data/canard/dev.rewrite.tsv \
  --output_text_pair dev.canard.rewrite.top1000.text_pair.txt \
  --output_id_pair dev.canard.rewrite.top1000.id_pair.txt

