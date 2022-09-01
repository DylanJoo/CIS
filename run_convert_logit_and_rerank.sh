####################################################################
# Submission candidates
# (0) Baseline (weak) monot5 msmarco large + rewritten query
# (1) Baseline (strong) monot5m msmarco large + rewritten query
# (2) conv.monot5 I:  Denoised conv.monot5 canard4ir + conversational query
# (3) conv.monot5 II:  Denoised conv.monot5m canard4ir + conversational query
####################################################################
TYPE=automatic.rewrite

##################################################################### 
# Conversational re-ranking
####################################################################
FOLDER=data/cast20/conv-monot5-pairs
for SCLIN_RUN in sclin.cqe sclin.t5-cqe sclin.t5;do
    TYPE=${SCLIN_RUN##*sclin.}

    MODEL=conv-monot5-large-denoise-canard4ir-10k 
    # note here we use denoise as our default setting instead of CQE's top3
    python3 tools/convert_logit_to_rerank.py \
      -flogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.flogits \
      -tlogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.tlogits \
      -score ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.scores \
      -runs runs/cast_result/result.${TYPE}.cast2020.eval.trec \
      -rerank_runs runs/cast20.${SCLIN_RUN}.top1000.conv.monot5.trec \
      -topk 1000 \
      --resoftmax &

    MODEL=conv-monot5m-large-canard4ir-10k
    python3 tools/convert_logit_to_rerank.py \
      -flogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.flogits \
      -tlogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.tlogits \
      -score ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.scores \
      -runs runs/cast_result/result.${TYPE}.cast2020.eval.trec \
      -rerank_runs runs/cast20.${SCLIN_RUN}.top1000.conv.monot5m.trec \
      -topk 1000 \
      --resoftmax &
done

##################################################################### 
# Standard re-ranking
####################################################################
FOLDER=data/cast20/monot5-pairs
for SCLIN_RUN in sclin.cqe sclin.t5-cqe sclin.t5;do
    TYPE=${SCLIN_RUN##*sclin.}

    MODEL=monot5-large-msmarco-100k 
    # note here we use denoise as our default setting instead of CQE's top3
    python3 tools/convert_logit_to_rerank.py \
      -flogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.flogits \
      -tlogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.tlogits \
      -score ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.scores \
      -runs runs/cast_result/result.${TYPE}.cast2020.eval.trec \
      -rerank_runs runs/cast20.${SCLIN_RUN}.top1000.monot5.trec \
      -topk 1000 \
      --resoftmax &

    MODEL=monot5m-large-msmarco-100k
    python3 tools/convert_logit_to_rerank.py \
      -flogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.flogits \
      -tlogits ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.tlogits \
      -score ${FOLDER}/${MODEL}/cast20.${SCLIN_RUN}.top1000.pred.scores \
      -runs runs/cast_result/result.${TYPE}.cast2020.eval.trec \
      -rerank_runs runs/cast20.${SCLIN_RUN}.top1000.monot5m.trec \
      -topk 1000 \
      --resoftmax &
done
