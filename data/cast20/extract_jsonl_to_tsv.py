import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_jsonl', default='2020_evaluation_topics_v1.0.jsonl')
    parser.add_argument('--output_tsv', default=None)
    parser.add_argument('--value', default='automatic_rewritten')
    args = parser.parse_args()

    if args.output_tsv is None:
        args.output_tsv = f"2020_{args.value}_evaluation_topics_v1.0.tsv"

    with open(args.evaluation_jsonl, 'r') as fin, \
            open(args.output_tsv, 'w') as fout:

        for line in fin:
            example = json.loads(line.strip())
            key = example['id']
            value = example[f'{args.value}']
            fout.write(f"{key}\t{value}\n")
