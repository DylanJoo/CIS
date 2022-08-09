DIR=convir_data_construction

# Rewrite retrieval with candidate from Rewrite (student)
python3 tools/convert_run_to_monot5.py \
  -run $DIR/runs/cast20.canard.train.rewrite.spr.top1000.trec \
  -corpus data/cast20/collections/  \
  -k 1000 \
  -q data/canard/train.rewrite.tsv \
  --output_text_pair $DIR/monot5-pairs/canard.train.student.top1000.text_pair \
  --output_id_pair $DIR/monot5-pairs/canard.train.student.top1000.id_pair

# Rewrite retrieval with candidate from Answer (teacher)
python3 tools/convert_run_to_monot5.py \
  -run $DIR/runs/cast20.canard.train.answer.spr.top1000.trec \
  -corpus data/cast20/collections/  \
  -k 1000 \
  -q data/canard/train.rewrite.tsv \
  --output_text_pair $DIR/monot5-pairs/canard.train.teacher.top1000.text_pair \
  --output_id_pair $DIR/monot5-pairs/canard.train.teacher.top1000.id_pair

