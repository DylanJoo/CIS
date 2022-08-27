FOLDER=data/cast20/conv-monot5-pairs
TYPE=automatic
Q_FORM=allq

MODEL=model-pn-top3-wndw-allq
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5.pntop3.wndwallq.trec \
  -topk 1000 \
  --resoftmax &

MODEL=model-pn-overlap-wndw-allq
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5.pnol.wndwallq.trec \
  -topk 1000 \
  --resoftmax &

MODEL=model-pn-top3-wndw-3
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5.pntop3.wndw3.trec \
  -topk 1000 \
  --resoftmax &

# MODEL=model-pn-overlap-wndw-3
# python3 tools/convert_logit_to_rerank.py \
#   -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.flogits \
#   -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.tlogits \
#   -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.scores \
#   -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
#   -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5.pnol.wndw3.trec \
#   -topk 1000 \
#   --resoftmax &

# multi
MODEL=model-pn-top3-wndw-allq-multi
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.flogits \
  -tlogits ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.tlogits \
  -score ${FOLDER}/${MODEL}/cast20.${TYPE}.rewrite.${Q_FORM}.top1000.pred.scores \
  -runs runs/cast20.${TYPE}.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.${TYPE}.rewrite.spr.top1000.conv.monot5.pntop3.wndwallq.multi.trec \
  -topk 1000 \
  --resoftmax &


FOLDER=data/cast20/monot5-pairs
# standard monot5
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/cast20.automatic.rewrite.top1000.pred.flogits \
  -tlogits ${FOLDER}/cast20.automatic.rewrite.top1000.pred.tlogits \
  -score ${FOLDER}/cast20.automatic.rewrite.top1000.pred.scores \
  -runs runs/cast20.automatic.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.automatic.rewrite.spr.top1000.monot5.trec \
  -topk 1000 \
  --resoftmax &

# standard monot5 with zero shot conv rerank
python3 tools/convert_logit_to_rerank.py \
  -flogits ${FOLDER}/cast20.automatic.rewrite.zeroshot.top1000.pred.flogits \
  -tlogits ${FOLDER}/cast20.automatic.rewrite.zeroshot.top1000.pred.tlogits \
  -score ${FOLDER}/cast20.automatic.rewrite.zeroshot.top1000.pred.scores \
  -runs runs/cast20.automatic.rewrite.spr.top1000.trec \
  -rerank_runs runs/cast20.automatic.rewrite.zeroshot.spr.top1000.monot5.trec \
  -topk 1000 \
  --resoftmax &

