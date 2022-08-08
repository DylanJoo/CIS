# automatic
python3 tools/convert_run_to_conv_monot5.py \
  -run data/cast20/y2_automatic_results_500.v1.0.run \
  -corpus data/cast20/collections/ \
  -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
  -output data/cast20/monot5-pairs/cast20.automatic.baseline.top500.jsonl

# manual
python3 tools/convert_run_to_conv_monot5.py \
  -run data/cast20/y2_manual_results_500.v1.0.run \
  -corpus data/cast20/collections/ \
  -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
  -output data/cast20/monot5-pairs/cast20.manual.baseline.top500.jsonl
