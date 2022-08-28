TYPE=automatic

# conv monot5
FOLDER=data/cast20/conv-monot5-pairs
MODEL=conv-monot5-base-canard4ir-10k
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5.trec \
  -topk 1000 \
  --resoftmax &

MODEL=conv-monot5-base-denoise-canard4ir-10k
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.denoise.conv.monot5.trec \
  -topk 1000 \
  --resoftmax &

MODEL=conv-monot5m-base-canard4ir-10k
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5m.trec \
  -topk 1000 \
  --resoftmax &

MODEL=conv-monot5m-large-canard4ir-10k
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5m.large.trec \
  -topk 1000 \
  --resoftmax &

# monot5
FOLDER=data/cast20/monot5-pairs
MODEL=monot5-base-msmarco-100k
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.monot5.trec \
  -topk 1000 \
  --resoftmax &

MODEL=monot5-base-msmarco-100k-zs
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.monot5.zeroshot.trec \
  -topk 1000 \
  --resoftmax &

MODEL=monot5m-large-msmarco-100k
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.monot5.large.trec \
  -topk 1000 \
  --resoftmax &
