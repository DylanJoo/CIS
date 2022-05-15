"""
First-step training for albert model (further pretraining)

Training functions for sentence highlighting, 
which includes two methods based on deep NLP pretrained models.

Backbone Models:
    (1) albert-base-v2

Methods:
    (1) Further pretraining: Passage ranking task adaptive finetunning

Packages requirments:
    - hugginface 
    - datasets 
"""
import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DefaultDataCollator,
)

from datasets import load_dataset, DatasetDict
from models import AlbertForCrossContextAwareTextRanking
from trainers import AlbertTrainer, AlbertTrainerForConvBatch

import os
os.environ["WANDB_DISABLED"] = "true"

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='albert-base-v2')
    model_type: Optional[str] = field(default='albert-base-v2')
    config_name: Optional[str] = field(default='albert-base-v2')
    tokenizer_name: Optional[str] = field(default='albert-base-v2')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Cutomized arguments
    pooler_type: str = field(default="cls")
    temp: float = field(default=0.05)
    num_labels: int = field(default=2)
    project_size: int = field(default=128)
    project_dropout_prob: int = field(default=None)

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default="data/train.jsonl")
    eval_file: Optional[str] = field(default="data/sample.jsonl")
    test_file: Optional[str] = field(default="data/sample.jsonl")
    max_q_seq_length: Optional[int] = field(default=64)
    max_p_seq_length: Optional[int] = field(default=128)
    max_c_seq_length: Optional[int] = field(default=128)
    pad_to_strategy: str = field(default="max_length")
    use_conversational_history: str = field(default="last_history", metadata={"help": "e.g. 'history', 'last_history'"})

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments. 
    output_dir: str = field(default='./models')
    seed: int = field(default=42)
    data_seed: int = field(default=None, metadata={"help": "for data sampler, set as the seed if None"})
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=100)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='steps')
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_ratio: float = field(default=0.1)
    warmup_steps: int = field(default=0)
    resume_from_checkpiint: Optional[str] = field(default=None)


def main():
    """
    (1) Prepare parser with the 3 types of arguments
        * Detailed argument parser by kwargs
    (2) Load the corresponding tokenizer and config 
    (3) Load the self-defined models
    (4)
    """

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # config and tokenizers
    # [TODO] Overwrite the initial argument from huggingface
    config_kwargs = {
            "num_labels": model_args.num_labels,
            "output_hidden_states": True
    }
    tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir, 
            "use_fast": model_args.use_fast_tokenizer
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    # model 
    model = AlbertForCrossContextAwareTextRanking.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config, 
            model_args=model_args,
    )

    # Dataset 
    def prepare_retrieval_pretraining(examples, context=None):

        size = len(examples['label'])
        features = tokenizer(
            examples['question'],
            max_length=data_args.max_q_seq_length,
            truncation=True,
            padding=data_args.pad_to_strategy,
        )
        features_passage = tokenizer(
            examples['passage'],
            max_length=data_args.max_p_seq_length,
            truncation=True,
            padding=data_args.pad_to_strategy,
        )
        features['query_input_ids'] = features['input_ids']
        features['query_attention_mask'] = features['attention_mask']
        features['query_token_type_ids'] = features['token_type_ids']
        features['passage_input_ids'] = features_passage['input_ids']
        features['passage_attention_mask'] = features_passage['attention_mask']
        features['passage_token_type_ids'] = features_passage['token_type_ids']
        features['ranking_label'] = examples['label']

        if context:
            features_context = tokenizer(
                examples[context],
                max_length=data_args.max_c_seq_length,
                truncation=True,
                padding=data_args.pad_to_strategy,
            )
            features['context_input_ids'] = features_context['input_ids']
            features['context_attention_mask'] = features_context['attention_mask']
            features['context_token_type_ids'] = features_context['token_type_ids']

            features_context = tokenizer(
                examples['utterance'],
                max_length=data_args.max_q_seq_length,
                truncation=True,
                padding=data_args.pad_to_strategy,
            )
            features['utterance_input_ids'] = features_context['input_ids']
            features['utterance_attention_mask'] = features_context['attention_mask']
            features['utterance_token_type_ids'] = features_context['token_type_ids']

        return features

    ## Loading form json
    dataset = DatasetDict.from_json({
        "train": data_args.train_file,
        "eval": data_args.eval_file
    })

    ## Preprocessing: training dataset and evaliatopm dataset
    dataset = dataset.map(
            function=prepare_retrieval_pretraining,
            fn_kwargs={'context': data_args.use_conversational_history},
            batched=True,
            remove_columns=['history', 'question', 'passage', 'utterance'],
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=not data_args.overwrite_cache,
    )
    dataset = dataset.remove_columns([
        'input_ids', 'attention_mask', 'token_type_ids', 'label', data_args.use_conversational_history
    ])

    # data collator (transform the datset into the training mini-batch)
    # [TODO] It should be the customized data collator
    data_collator = DefaultDataCollator(
            return_tensors="pt",
    )

    # Trainer
    # trainer = AlbertTrainer(
    trainer = AlbertTrainerForConvBatch(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'],
            data_collator=data_collator
    )
    
    # ***** strat training *****
    model_path = None #[TODO] parsing the argument model_args.model_name_or_path 
    results = trainer.train(model_path=model_path)

    return results

if __name__ == '__main__':
    main()
