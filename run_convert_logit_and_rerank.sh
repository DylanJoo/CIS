# folder=re-ranking/train.canard.answer.monot5.pred
folder=$1
python3 tools/convert_logit_to_rerank.py \
  -flogits ${folder}/flogits \
  -tlogits ${folder}/tlogits \
  -score ${folder}/scores \
  -runs ${folder/monot5.pred/top1000.id_pair.txt} \
  -rerank_runs re-ranking/runs/${folder##*/}.top1000.rerank.trec \
  -topk 1000 \
  --resoftmax \
  --trec
