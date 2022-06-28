"""
The specified corpus in 'collection-path' is basically the wikipedia dumps, 
which are parsed and preprocessed by the authores in OR-ConvQA paper,

* Title: Open-Retrieval Conversational Question Answering
* link : https://arxiv.org/pdf/2005.11364.pdf
"""
import json
import os
import argparse

def convert_collection(args):
    print('Converting collection...')
    file_index = 0
    with open(args.collection_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            doc_dict = json.loads(line.strip())
            doc_id, doc_text = doc_dict['id'], doc_dict['text']

            if i % args.max_docs_per_file == 0:
                if i > 0:
                    output_jsonl_file.close()
                output_path = os.path.join(args.output_folder, 'ORCONVQA{:02d}.json'.format(file_index))
                output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                file_index += 1
            output_dict = {'id': f'ORCONVQA_{doc_id}', 'contents': doc_text}
            output_jsonl_file.write(json.dumps(output_dict) + '\n')

            if i % 100000 == 0:
                print(f'Converted {i:,} docs, writing into file {file_index}')

    output_jsonl_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all_blocks.txt collection into jsonl files for Anserini.')
    parser.add_argument('--collection-path', required=True, help='Path to wiki collections of orconvqa.')
    parser.add_argument('--output-folder', required=True, help='Output folder.')
    parser.add_argument('--max-docs-per-file', default=1000000, type=int,
                        help='Maximum number of documents in each jsonl file.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    convert_collection(args)
    print('Done!')
