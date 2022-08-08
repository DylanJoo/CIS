# train
python3 tools/extract_canard_query.py \
  -canard data/canard/train.json \
  -full \
  -quac data/quac \
  -col2 Question \
  -col2 Rewrite \
  -col2 Answer \
  -col2 History
 
# dev
python3 tools/extract_canard_query.py \
  -canard data/canard/dev.json \
  -full \
  -quac data/quac \
  -col2 Question \
  -col2 Rewrite \
  -col2 Answer \
  -col2 History
 
