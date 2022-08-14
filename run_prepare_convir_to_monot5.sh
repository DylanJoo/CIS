DIR=convir_data_construction

# Rewrite retrieval with candidate from Rewrite (student)
python3 tools/convert_run_to_monot5.py \
  -run $DIR/runs/cast20.canard.train.rewrite.spr.top1000.trec \
  -topic data/canard/train.queries.jsonl \
  -corpus data/cast20/collections/  \
  --output_text_pair $DIR/dataset_construction/canard.train.student.top1000.text_pair2 \
  --output_id_pair $DIR/dataset_construction/canard.train.student.top1000.id_pair &
  

# Rewrite retrieval with candidate from Answer (teacher)
python3 tools/convert_run_to_monot5.py \
  -run $DIR/runs/cast20.canard.train.answer.spr.top1000.trec \
  -topic data/canard/train.queries.jsonl \
  -corpus data/cast20/collections/  \
  --output_text_pair $DIR/dataset_construction/canard.train.teacher.top1000.text_pair2 \
  --output_id_pair $DIR/dataset_construction/canard.train.teacher.top1000.id_pair

