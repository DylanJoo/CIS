folder=data/cast20/monot5-pairs

python3 tools/convert_logit_to_rerank.py \
  -flogits ${folder}/cast20.automatic.baseline.top500.pred.flogits \
  -tlogits ${folder}/cast20.automatic.baseline.top500.pred.tlogits \
  -score ${folder}/cast20.automatic.baseline.top500.pred.scores \
  -runs data/cast20/y2_automatic_results_500.v1.0.run \
  -rerank_runs runs/cast20.automatic.baseline.top500.monot5.trec \
  -topk 1000 \
  --resoftmax &

python3 tools/convert_logit_to_rerank.py \
  -flogits ${folder}/cast20.manual.baseline.top500.pred.flogits \
  -tlogits ${folder}/cast20.manual.baseline.top500.pred.tlogits \
  -score ${folder}/cast20.manual.baseline.top500.pred.scores \
  -runs data/cast20/y2_manual_results_500.v1.0.run \
  -rerank_runs runs/cast20.manual.baseline.top500.monot5.trec \
  -topk 1000 \
  --resoftmax
