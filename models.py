import os
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AlbertModel, AlbertPreTrainedModel
from transformers.file_utils import WEIGHTS_NAME, cached_path

class AlbertForTextRanking(AlbertPreTrainedModel):
    R"""
    The text ranking model using albert with siamese network.
    """
    def __init__(self, config, **model_kargs):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config
        self.model_args = model_kargs["model_args"]

        self.query_encoder = AlbertModel(config)
        self.passage_encoder = AlbertModel(config)
        self.query_project = nn.Linear(config.hidden_size, self.model_args.project_size)
        self.passage_project = nn.Linear(config.hidden_size, self.model_args.project_size)

        project_dropout = (
                self.model_args.project_dropout_prob if self.model_args.project_dropout_prob is not None \
                        else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(project_dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.post_init()

    def forward(self, 
                query_input_ids=None, passage_input_ids=None, 
                query_attention_mask=None, passage_attention_mask=None,
                query_token_type_ids=None, passage_token_type_ids=None,
                ranking_label=None,):
        f"""
        Bi-encoder architecture with two albert, for query and passage respectively
        first obtain the representations of each, then compute the inner product
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # (1) Retrieval pre-training (text ranking task) ...
        # (2) Fine-tunning query encoder 

        # Getting the reduced vector (distinct to hidden size)
        if query_input_ids is not None:
            query_outputs = self.forward_query(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    token_type_ids=query_token_type_ids,
            )
            query_reprs = query_outputs["repr"]

        if passage_input_ids is not None:
            passage_outputs = self.forward_passage(
                    input_ids=passage_input_ids,
                    attention_mask=passage_attention_mask,
                    token_type_ids=passage_token_type_ids,
            )
            passage_reprs = passage_outputs["repr"]
            passage_reprs_t = passage_reprs.transpose(0, 1) # transpose it for further matmul
        
        ranking_logit = torch.matmul(query_reprs, passage_reprs_t)
        ranking_label = torch.arange(
                query_reprs.size(0),
                device=query_outputs.device, 
        )
        ranking_loss_fct = CrossEntropyLoss()
        ranking_loss = ranking_loss_fct(ranking_logit, ranking_label)

        # Return the hidden state and attention from query encoder
        return SequenceClassifierOutput(
            loss=ranking_loss,
            logits=ranking_logit,
            hidden_states=query_outputs.hidden_states,
            attentions=query_outputs.attentions,
        )

    def forward_query(self,
                      input_ids=None,
                      attention_mask=None,
                      token_type_ids=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      labels=None,
                      sentence_order_label=None,
                      output_attentions=None,
                      output_hidden_states=None,
                      return_dict=None,):

        outputs = self.query_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embed = outputs[1] # i.e. pooled output
        representation = self.query_project(self.dropout(sequence_embeds))
        # loss = None

        return representation

    def forward_passage(self,
                        input_ids=None,
                        attention_mask=None,
                        token_type_ids=None,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
                        labels=None,
                        sentence_order_label=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=None,):

        outputs = self.passage_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embed = outputs[1] # i.e. pooled output
        representation = self.passage_project(self.dropout(sequence_embeds))
        # loss = None

        return representation


    def inference(self, inputs):
        pass

        # with torch.no_grad():
        #     outputs = self.forward(**inputs)
        #     probabilities = self.softmax(self.tokens_clf(outputs.hidden_states[-1]))
        #     predictions = torch.argmax(probabilities, dim=-1)
        #
        #     # active filtering
        #     active_tokens = inputs['attention_mask'] == 1
        #     active_predictions = torch.where(
        #         active_tokens,
        #         predictions,
        #         torch.tensor(-1).type_as(predictions)
        #     )
        #     return {"probabilities": probabilities[:, :, 1].detach(), # shape: (batch, length)
        #             "active_predictions": predictions.detach(),
        #             "active_tokens": active_tokens}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if pretrained_model_name_or_path is not None and (
                "albert" in pretrained_model_name_or_path and "v2" in pretrained_model_name_or_path):
            logger.warning("There is currently an upstream reproducibility issue with ALBERT v2 models. Please see " +
                           "https://github.com/google-research/google-research/issues/119 for more information.")

        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                force_download=force_download,
                proxies=proxies,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                # Pytorch
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                    is_shared = True
                else:
                    raise EnvironmentError("Error no file found in directory {} or \
                            `from_tf` set to False".format(pretrained_model_name_or_path))
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert from_tf, "We found a TensorFlow checkpoint at {}, \
                        please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index")

                archive_file = pretrained_model_name_or_path + ".index"
            

  
            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download,
                                                    proxies=proxies)
            except EnvironmentError:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                            archive_file)
                else:
                    msg = "Model name '{}' was not found in model name list ({}). " \
                        "We assumed '{}' was a path or url to model weight files named one of {} but " \
                        "couldn't find any such file at this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(cls.pretrained_model_archive_map.keys()),
                            archive_file,
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
                raise EnvironmentError(msg)
                
            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model
                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError as e:
                    logger.error("Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
                    raise e
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if key == 'lm_head.decoder.weight':
                    new_key = 'lm_head.weight'
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            # print('orig state dict', state_dict.keys(), len(state_dict))
            customized_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                k_split = k.split('.')
                if k_split[0] == 'bert':
                    k_split[0] = 'query_encoder'
                    customized_state_dict['.'.join(k_split)] = v
                    k_split[0] = 'passage_encoder'
                    customized_state_dict['.'.join(k_split)] = v
                    
            if len(customized_state_dict) == 0:
                # loading from our trained model
                state_dict = state_dict.copy()
                # print('using orig state dict', state_dict.keys())
            else:
                # loading from original bert model
                state_dict = customized_state_dict.copy()
                # print('using custome state dict', state_dict.keys())
            
            # print('modified state dict', state_dict.keys(), len(state_dict))
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
#             if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#                 start_prefix = cls.base_model_prefix + '.'
#             if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#                 model_to_load = getattr(model, cls.base_model_prefix)

#             load(model_to_load, prefix=start_prefix)
            load(model_to_load, prefix='')
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                model.__class__.__name__, "\n\t".join(error_msgs)))

        model.tie_weights()  # make sure word embedding weights are still tied if needed

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model
