folder=convir_data_construction

python3 tools/convert_logit_to_rerank.py \
  -flogits ${folder}/dataset_construction/teacher.flogits \
  -tlogits ${folder}/dataset_construction/teacher.tlogits \
  -score ${folder}/dataset_construction/teacher.scores \
  -runs ${folder}/runs/cast20.canard.train.answer.spr.top1000.trec \
  -rerank_runs ${folder}/runs/cast20.canard.train.teacher.top1000.monot5.trec \
  -topk 1000 \
  --resoftmax &

python3 tools/convert_logit_to_rerank.py \
  -flogits ${folder}/dataset_construction/student.flogits \
  -tlogits ${folder}/dataset_construction/student.tlogits \
  -score ${folder}/dataset_construction/student.scores \
  -runs ${folder}/runs/cast20.canard.train.rewrite.spr.top1000.trec \
  -rerank_runs ${folder}/runs/cast20.canard.train.student.top1000.monot5.trec \
  -topk 1000 \
  --resoftmax
