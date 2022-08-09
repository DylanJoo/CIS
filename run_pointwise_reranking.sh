# monot5
# automatic
python3 rerank/rerank.py \
    --jsonl_pair data/cast20/monot5-pairs/cast20.automatic.baseline.top500.jsonl \
    --input_trec data/cast20/y2_automatic_results_500.v1.0.run \
    --output_trec runs/cast20.automatic.eval.topics.baseline.top500.monot5.base.trec \
    --model_name_or_path 'castorini/monot5-base-msmarco' \
    --batch_size 8 \
    --max_q_seq_length 128 \
    --max_p_seq_length 384 \
    --gpu 0 \
    --prefix monoT5

# manual
python3 rerank/rerank.py \
    --jsonl_pair data/cast20/monot5-pairs/cast20.manual.baseline.top500.jsonl \
    --input_trec data/cast20/y2_manual_results_500.v1.0.run \
    --output_trec runs/cast20.manual.eval.topics.baseline.top500.monot5.base.trec \
    --model_name_or_path 'castorini/monot5-base-msmarco' \
    --batch_size 8 \
    --max_q_seq_length 128 \
    --max_p_seq_length 384 \
    --gpu 0 \
    --prefix monoT5


# # conv monot5
# # automatic
# python3 rerank/rerank.py \
#     --jsonl_pair data/cast20/monot5-pairs/cast20.automatic.baseline.top500.jsonl \
#     --input_trec data/cast20/y2_automatic_results_500.v1.0.run \
#     --output_trec runs/cast20.automatic.eval.topics.baseline.top500.conv.monot5.small.trec \
#     --model_name_or_path rerank/checkpoints/conv.monot5/checkpoint-100000/ \
#     --batch_size 8 \
#     --max_q_seq_length 128 \
#     --max_p_seq_length 384 \
#     --gpu 0 \
#     --prefix conv.monoT5
#
# # manual
# python3 rerank/rerank.py \
#     --jsonl_pair data/cast20/monot5-pairs/cast20.manual.baseline.top500.jsonl \
#     --input_trec data/cast20/y2_manual_results_500.v1.0.run \
#     --output_trec runs/cast20.manual.eval.topics.baseline.top500.conv.monot5.small.trec \
#     --model_name_or_path rerank/checkpoints/conv.monot5/checkpoint-100000/ \
#     --batch_size 8 \
#     --max_q_seq_length 128 \
#     --max_p_seq_length 384 \
#     --gpu 0 \
#     --prefix conv.monoT5
