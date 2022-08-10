# train
python3 tools/extract_canard_query.py \
  -canard data/canard/train.json \
  -output data/canard/train.queries.jsonl 
 
# dev
python3 tools/extract_canard_query.py \
  -canard data/canard/dev.json \
  -output data/canard/dev.queries.jsonl 
