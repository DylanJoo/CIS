# text
# automatic
# python3 tools/convert_run_to_monot5.py \
#   -run data/cast20/y2_automatic_results_500.v1.0.run \
#   -corpus data/cast20/collections/ \
#   -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
#   --output_text_pair data/cast20/monot5-pairs/cast20.automatic.baseline.top500.text_pairs.txt \
#   --output_id_pair data/cast20/monot5-pairs/cast20.automatic.baseline.top500.id_pairs.txt
#
# # manual
# python3 tools/convert_run_to_monot5.py \
#   -run data/cast20/y2_manual_results_500.v1.0.run \
#   -corpus data/cast20/collections/ \
#   -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
#   --output_text_pair data/cast20/monot5-pairs/cast20.manual.baseline.top500.text_pairs.txt \
#   --output_id_pair data/cast20/monot5-pairs/cast20.manual.baseline.top500.id_pairs.txt

# automatic (user context)
python3 tools/convert_run_to_monot5.py \
  -run data/cast20/y2_automatic_results_500.v1.0.run \
  -corpus data/cast20/collections/ \
  -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
  --output_text_pair data/cast20/conv-monot5-pairs/cast20.automatic.baseline.top500.text_pairs.txt \
  --output_id_pair data/cast20/conv-monot5-pairs/cast20.automatic.baseline.top500.id_pairs.txt \
  --use_context 3 

# manual (user context)
python3 tools/convert_run_to_monot5.py \
  -run data/cast20/y2_manual_results_500.v1.0.run \
  -corpus data/cast20/collections/ \
  -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
  --output_text_pair data/cast20/conv-monot5-pairs/cast20.manual.baseline.top500.text_pairs.txt \
  --output_id_pair data/cast20/conv-monot5-pairs/cast20.manual.baseline.top500.id_pairs.txt \
  --use_context 3 

