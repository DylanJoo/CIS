# text
# automatic
python3 tools/convert_run_to_monot5.py \
  -run runs/cast20.automatic.rewrite.spr.top1000.trec \
  -corpus data/cast20/collections/ \
  -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
  --output_text_pair data/cast20/monot5-pairs/cast20.automatic.rewrite.top1000.monot5.text_pairs.txt \
  --output_id_pair data/cast20/monot5-pairs/cast20.automatic.rewrite.top1000.monot5.id_pairs.txt &

# manual
python3 tools/convert_run_to_monot5.py \
  -run runs/cast20.manual.rewrite.spr.top1000.trec \
  -corpus data/cast20/collections/ \
  -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
  --output_text_pair data/cast20/monot5-pairs/cast20.manual.rewrite.top1000.monot5..text_pairs.txt \
  --output_id_pair data/cast20/monot5-pairs/cast20.manual.rewrite.top1000.monot5.id_pairs.txt \
  --use_manual 

