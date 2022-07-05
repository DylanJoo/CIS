import os
from typing import Optional, Union, Dict, Any
import collections
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AlbertModel, AlbertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from transformers.utils import (
        WEIGHTS_NAME, 
        cached_path, 
        is_offline_mode,
        is_remote_url,
        hf_bucket_url,
        logging
)
from transformers.modeling_utils import (
        no_init_weights,
        load_state_dict
)

logger = logging.get_logger(__name__)

class AlbertForTextRanking(AlbertPreTrainedModel):
    R"""
    The text ranking model using albert with siamese network.
    """
    def __init__(self, config, **model_kargs):
        super(AlbertForTextRanking, self).__init__(config)
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

    def inference(self, inputs):

        with torch.no_grad():
            passage_reprs, _ = self.forward_passage(
                    intput_ids=inputs['passage_input_ids'],
                    attention_mask=passage_attention_mask,
                    token_type_ids=passage_token_type_ids
            )
        return passage_reprs


    def forward(self, 
                query_input_ids=None, passage_input_ids=None, 
                query_attention_mask=None, passage_attention_mask=None,
                query_token_type_ids=None, passage_token_type_ids=None,
                ranking_label=None, return_dict=None):
        f"""
        Bi-encoder architecture with two albert, for query and passage respectively
        first obtain the representations of each, then compute the inner product
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # (1) Retrieval pre-training (text ranking task) ...
        # (2) Fine-tunning query encoder 

        # Getting the reduced vector (distinct to hidden size)
        if query_input_ids is not None:
            query_reprs, query_outputs = self.forward_query(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    token_type_ids=query_token_type_ids,
            )

        if passage_input_ids is not None:
            passage_reprs, passage_outputs = self.forward_passage(
                    input_ids=passage_input_ids,
                    attention_mask=passage_attention_mask,
                    token_type_ids=passage_token_type_ids,
            )
            passage_reprs_t = passage_reprs.transpose(0, 1) # transpose it for further matmul
        
        ranking_logit = torch.matmul(query_reprs, passage_reprs_t)
        ranking_label = torch.arange(
                query_reprs.size(0),
                device=self.device, 
                dtype=ranking_label.dtype
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
                      output_hidden_states=None,):

        outputs = self.query_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=self.config.use_return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embeds = outputs[1] # i.e. pooled output
        representation = self.query_project(self.dropout(sequence_embeds))
        outputs["repr"] = representation

        return representation, outputs

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
                        output_hidden_states=None,):

        outputs = self.passage_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=self.config.use_return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embeds = outputs[1] # i.e. pooled output
        representation = self.passage_project(self.dropout(sequence_embeds))
        outputs["repr"] = representation

        return representation, outputs


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

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.
        Args:
            inputs (`dict`): The model inputs.
        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        # For the biencoder, separate to input into query's and passage's tokens
        try:
            return input_dict['query_input_ids'].numel() + input_dict['passage_input_ids'].numel()
        except:
            if self.main_input_name in input_dict:
                return input_dict[self.main_input_name].numel()
            elif "estimate_tokens" not in self.warnings_issued:
                logger.warning(
                    "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
                )
                self.warnings_issued["estimate_tokens"] = True
            return 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                        "weights."
                    )
                elif os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or "
                        f"{FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                else:
                    filename = WEIGHTS_NAME

                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=filename,
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                if filename == WEIGHTS_NAME:
                    try:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        archive_file = hf_bucket_url(
                            pretrained_model_name_or_path,
                            filename=WEIGHTS_INDEX_NAME,
                            revision=revision,
                            mirror=mirror,
                        )
                        resolved_archive_file = cached_path(
                            archive_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            resume_download=resume_download,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            user_agent=user_agent,
                        )
                        is_sharded = True
                    except EntryNotFoundError:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "mirror": mirror,
                            "proxies": proxies,
                            "use_auth_token": use_auth_token,
                        }
                        if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                                "weights."
                            )
                        elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                                "weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}, "
                                f"{TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                            )
                else:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                    )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached "
                    f"files and it looks like {pretrained_model_name_or_path} is not the path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}.\n"
                    "Checkout your internet connection or see how to run the library in offline mode at "
                    "'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}."
                )

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                mirror=mirror,
            )

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)
            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry - we assume all weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None
            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            torch_dtype = next(iter(state_dict.values())).dtype
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file)
                            torch_dtype = next(iter(one_state_dict.values())).dtype
                            del one_state_dict  # free CPU memory
                    else:
                        raise ValueError(
                            f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            if low_cpu_mem_usage:
                # save the keys
                if is_sharded:
                    loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
                else:
                    loaded_state_dict_keys = [k for k in state_dict.keys()]
                    del state_dict  # free CPU memory - will reload again later

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                with no_init_weights(_enable=_fast_init):
                    model = cls(config, *model_args, **model_kwargs)
        else:
            with no_init_weights(_enable=_fast_init):
                model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

        # [BIENCODER] Make the original model into two separated encoders.
        customized_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            k_split = k.split('.')
            if 'bert' in k_split[0]:
                k_split[0] = 'query_encoder'
                customized_state_dict['.'.join(k_split)] = v
                k_split[0] = 'passage_encoder'
                customized_state_dict['.'.join(k_split)] = v
        
        ## Check if the fine-tuned one
        if len(customized_state_dict) == 0:
            state_dict = state_dict.copy()
        else:
            state_dict = customized_state_dict.copy()


        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        elif from_flax:
            try:
                from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see "
                    "https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions."
                )
                raise
        elif from_pt:

            if low_cpu_mem_usage:
                cls._load_pretrained_model_low_mem(model, loaded_state_dict_keys, resolved_archive_file)
            else:
                model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                    model,
                    state_dict,
                    resolved_archive_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    sharded_metadata=sharded_metadata,
                    _fast_init=_fast_init,
                )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model


class AlbertForCrossContextAwareTextRanking(AlbertPreTrainedModel):
    def __init__(self, config, **model_kargs):
        super(AlbertForCrossContextAwareTextRanking, self).__init__(config)
        # self.num_labels = config.num_labels
        self.config = config
        self.model_args = model_kargs["model_args"]

        self.query_encoder = AlbertModel(config)
        self.passage_encoder = AlbertModel(config)
        self.context_gru = nn.GRU(self.model_args.project_size, self.model_args.project_size)
        self.query_project = nn.Linear(config.hidden_size, self.model_args.project_size)
        self.passage_project = nn.Linear(config.hidden_size, self.model_args.project_size)

        project_dropout = (
                self.model_args.project_dropout_prob if self.model_args.project_dropout_prob is not None \
                        else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(project_dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.post_init()

    def forward_context(self, inputs_embeds=None):

        # first go throught the cross encoder of albert
        inputs_embeds = inputs_embeds[:, None, :] # (B(seq-of-turn), 1(batch in gru), H)
        context_reprs_recurrent, _ = self.context_gru(inputs_embeds) 

        return torch.squeeze(context_reprs_recurrent) # (B, H)

    def forward(self, 
                query_input_ids=None, query_attention_mask=None, query_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None,
                utterance_input_ids=None, utterance_attention_mask=None, utterance_token_type_ids=None,
                context_input_ids=None, context_attention_mask=None, context_token_type_ids=None,
                ranking_label=None, return_dict=None):

        utterance_reprs, _ = self.forward_query(
                input_ids=utterance_input_ids,
                attention_mask=utterance_attention_mask,
                token_type_ids=utterance_token_type_ids,
        )
        context_reprs, _ = self.forward_query(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                token_type_ids=context_token_type_ids,
        )
        context_reprs_rnn = self.forward_context(
                inputs_embeds=context_reprs, # (B(seq-of-turn) x H)
        )
        # Enhanced query representation of utterance and context
        query_reprs_rnn = context_reprs_rnn + utterance_reprs

        # The rewrite representation
        if query_input_ids is not None:
            query_reprs, query_outputs = self.forward_query(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    token_type_ids=query_token_type_ids,
            )
        if passage_input_ids is not None:
            passage_reprs, passage_outputs = self.forward_passage(
                    input_ids=passage_input_ids,
                    attention_mask=passage_attention_mask,
                    token_type_ids=passage_token_type_ids,
            )
            passage_reprs_t = passage_reprs.transpose(0, 1)


        ranking_loss = 0
        conversation_loss = 0

        ranking_logit = torch.matmul(query_reprs_rnn, passage_reprs_t)
        ranking_label[ranking_label == 1] = torch.arange(
                sum(ranking_label == 1), # query_reprs.size(0),
                device=self.device, 
                dtype=ranking_label.dtype
        )
        # compensate example removed
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
                      output_hidden_states=None,):

        outputs = self.query_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=self.config.use_return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embeds = outputs[1] # i.e. pooled output
        representation = self.query_project(self.dropout(sequence_embeds))
        outputs["repr"] = representation

        return representation, outputs

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
                        output_hidden_states=None,):

        outputs = self.passage_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=self.config.use_return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embeds = outputs[1] # i.e. pooled output
        representation = self.passage_project(self.dropout(sequence_embeds))
        outputs["repr"] = representation

        return representation, outputs


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

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.
        Args:
            inputs (`dict`): The model inputs.
        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        # For the biencoder, separate to input into query's and passage's tokens
        try:
            return input_dict['query_input_ids'].numel() + input_dict['passage_input_ids'].numel()
        except:
            if self.main_input_name in input_dict:
                return input_dict[self.main_input_name].numel()
            elif "estimate_tokens" not in self.warnings_issued:
                logger.warning(
                    "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
                )
                self.warnings_issued["estimate_tokens"] = True
            return 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                        "weights."
                    )
                elif os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or "
                        f"{FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                else:
                    filename = WEIGHTS_NAME

                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=filename,
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                if filename == WEIGHTS_NAME:
                    try:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        archive_file = hf_bucket_url(
                            pretrained_model_name_or_path,
                            filename=WEIGHTS_INDEX_NAME,
                            revision=revision,
                            mirror=mirror,
                        )
                        resolved_archive_file = cached_path(
                            archive_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            resume_download=resume_download,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            user_agent=user_agent,
                        )
                        is_sharded = True
                    except EntryNotFoundError:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "mirror": mirror,
                            "proxies": proxies,
                            "use_auth_token": use_auth_token,
                        }
                        if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                                "weights."
                            )
                        elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                                "weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}, "
                                f"{TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                            )
                else:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                    )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached "
                    f"files and it looks like {pretrained_model_name_or_path} is not the path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}.\n"
                    "Checkout your internet connection or see how to run the library in offline mode at "
                    "'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}."
                )

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                mirror=mirror,
            )

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)
            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry - we assume all weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None
            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            torch_dtype = next(iter(state_dict.values())).dtype
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file)
                            torch_dtype = next(iter(one_state_dict.values())).dtype
                            del one_state_dict  # free CPU memory
                    else:
                        raise ValueError(
                            f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            if low_cpu_mem_usage:
                # save the keys
                if is_sharded:
                    loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
                else:
                    loaded_state_dict_keys = [k for k in state_dict.keys()]
                    del state_dict  # free CPU memory - will reload again later

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                with no_init_weights(_enable=_fast_init):
                    model = cls(config, *model_args, **model_kwargs)
        else:
            with no_init_weights(_enable=_fast_init):
                model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

        # [BIENCODER] Make the original model into two separated encoders.
        customized_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            k_split = k.split('.')
            if 'bert' in k_split[0]:
                k_split[0] = 'query_encoder'
                customized_state_dict['.'.join(k_split)] = v
                k_split[0] = 'passage_encoder'
                customized_state_dict['.'.join(k_split)] = v
        
        ## Check if the fine-tuned one
        if len(customized_state_dict) == 0:
            state_dict = state_dict.copy()
        else:
            state_dict = customized_state_dict.copy()


        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        elif from_flax:
            try:
                from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see "
                    "https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions."
                )
                raise
        elif from_pt:

            if low_cpu_mem_usage:
                cls._load_pretrained_model_low_mem(model, loaded_state_dict_keys, resolved_archive_file)
            else:
                model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                    model,
                    state_dict,
                    resolved_archive_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    sharded_metadata=sharded_metadata,
                    _fast_init=_fast_init,
                )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model


class AlbertForDualContextAwareTextRanking(AlbertPreTrainedModel):
    def __init__(self, config, **model_kargs):
        super(AlbertForDualContextAwareTextRanking, self).__init__(config)
        # self.num_labels = config.num_labels
        self.config = config
        self.model_args = model_kargs["model_args"]

        self.query_encoder = AlbertModel(config)
        self.passage_encoder = AlbertModel(config)
        self.context_gru = nn.GRU(self.model_args.project_size, self.model_args.project_size)
        self.query_project = nn.Linear(config.hidden_size, self.model_args.project_size)
        self.passage_project = nn.Linear(config.hidden_size, self.model_args.project_size)
        self.context_project = nn.Linear(2*self.model_args.project_size, self.model_args.project_size)

        project_dropout = (
                self.model_args.project_dropout_prob if self.model_args.project_dropout_prob is not None \
                        else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(project_dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.post_init()

    def forward_context(self, inputs_embeds=None):

        # first go throught the cross encoder of albert
        inputs_embeds = inputs_embeds[:, None, :] # (B(seq-of-turn), 1(batch in gru), H)
        context_reprs_recurrent, _ = self.context_gru(inputs_embeds) 

        return torch.squeeze(context_reprs_recurrent) # (B, H)

    def forward(self, 
                query_input_ids=None, query_attention_mask=None, query_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None,
                utterance_input_ids=None, utterance_attention_mask=None, utterance_token_type_ids=None,
                context_input_ids=None, context_attention_mask=None, context_token_type_ids=None,
                context_q_input_ids=None, context_q_attention_mask=None, context_q_token_type_ids=None,
                context_r_input_ids=None, context_r_attention_mask=None, context_r_token_type_ids=None,
                ranking_label=None, return_dict=None):

        # The rewrite representation
        if query_input_ids is not None:
            query_reprs, query_outputs = self.forward_query(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    token_type_ids=query_token_type_ids,
            )
        if passage_input_ids is not None:
            passage_reprs, passage_outputs = self.forward_passage(
                    input_ids=passage_input_ids,
                    attention_mask=passage_attention_mask,
                    token_type_ids=passage_token_type_ids,
            )
            passage_reprs_t = passage_reprs.transpose(0, 1)
        if context_input_ids is not None:
            context_reprs, _ = self.forward_query(
                    input_ids=context_q_input_ids,
                    attention_mask=context_q_attention_mask,
                    token_type_ids=context_q_token_type_ids,
            )

        # Context encoders
        utterance_reprs, _ = self.forward_query(
                input_ids=utterance_input_ids,
                attention_mask=utterance_attention_mask,
                token_type_ids=utterance_token_type_ids,
        )
        context_q_reprs, _ = self.forward_query(
                input_ids=context_q_input_ids,
                attention_mask=context_q_attention_mask,
                token_type_ids=context_q_token_type_ids,
        )
        context_r_reprs, _ = self.forward_passage(
                input_ids=context_r_input_ids,
                attention_mask=context_r_attention_mask,
                token_type_ids=context_r_token_type_ids,
        )
        context_reprs = self.context_project(
                self.dropout(torch.cat((context_q_reprs, context_r_reprs), dim=1))
        )
        context_reprs_rnn = self.forward_context(
                inputs_embeds=context_reprs, # (B(seq-of-turn) x H)
        )

        # Enhanced query representation of utterance and context
        query_reprs_rnn = context_reprs_rnn + utterance_reprs
        ranking_loss = 0

        ranking_logit = torch.matmul(query_reprs_rnn, passage_reprs_t)
        ranking_label[ranking_label == 1] = torch.arange(
                sum(ranking_label == 1), # query_reprs.size(0),
                device=self.device, 
                dtype=ranking_label.dtype
        )
        # compensate example removed
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
                      output_hidden_states=None,):

        outputs = self.query_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=self.config.use_return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embeds = outputs[1] # i.e. pooled output
        representation = self.query_project(self.dropout(sequence_embeds))
        outputs["repr"] = representation

        return representation, outputs

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
                        output_hidden_states=None,):

        outputs = self.passage_encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=self.config.use_return_dict,
        )
        
        # tokens_embeds = outputs[0] # i.e. last hidden layer of all tokens
        sequence_embeds = outputs[1] # i.e. pooled output
        representation = self.passage_project(self.dropout(sequence_embeds))
        outputs["repr"] = representation

        return representation, outputs


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

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.
        Args:
            inputs (`dict`): The model inputs.
        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        # For the biencoder, separate to input into query's and passage's tokens
        try:
            return input_dict['query_input_ids'].numel() + input_dict['passage_input_ids'].numel()
        except:
            if self.main_input_name in input_dict:
                return input_dict[self.main_input_name].numel()
            elif "estimate_tokens" not in self.warnings_issued:
                logger.warning(
                    "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
                )
                self.warnings_issued["estimate_tokens"] = True
            return 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                        "weights."
                    )
                elif os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or "
                        f"{FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                else:
                    filename = WEIGHTS_NAME

                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=filename,
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                if filename == WEIGHTS_NAME:
                    try:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        archive_file = hf_bucket_url(
                            pretrained_model_name_or_path,
                            filename=WEIGHTS_INDEX_NAME,
                            revision=revision,
                            mirror=mirror,
                        )
                        resolved_archive_file = cached_path(
                            archive_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            resume_download=resume_download,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            user_agent=user_agent,
                        )
                        is_sharded = True
                    except EntryNotFoundError:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "mirror": mirror,
                            "proxies": proxies,
                            "use_auth_token": use_auth_token,
                        }
                        if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                                "weights."
                            )
                        elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                                "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                                "weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}, "
                                f"{TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                            )
                else:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                    )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached "
                    f"files and it looks like {pretrained_model_name_or_path} is not the path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}.\n"
                    "Checkout your internet connection or see how to run the library in offline mode at "
                    "'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}."
                )

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                mirror=mirror,
            )

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)
            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry - we assume all weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None
            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            torch_dtype = next(iter(state_dict.values())).dtype
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file)
                            torch_dtype = next(iter(one_state_dict.values())).dtype
                            del one_state_dict  # free CPU memory
                    else:
                        raise ValueError(
                            f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            if low_cpu_mem_usage:
                # save the keys
                if is_sharded:
                    loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
                else:
                    loaded_state_dict_keys = [k for k in state_dict.keys()]
                    del state_dict  # free CPU memory - will reload again later

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                with no_init_weights(_enable=_fast_init):
                    model = cls(config, *model_args, **model_kwargs)
        else:
            with no_init_weights(_enable=_fast_init):
                model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

        # [BIENCODER] Make the original model into two separated encoders.
        customized_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            k_split = k.split('.')
            if 'bert' in k_split[0]:
                k_split[0] = 'query_encoder'
                customized_state_dict['.'.join(k_split)] = v
                k_split[0] = 'passage_encoder'
                customized_state_dict['.'.join(k_split)] = v
        
        ## Check if the fine-tuned one
        if len(customized_state_dict) == 0:
            state_dict = state_dict.copy()
        else:
            state_dict = customized_state_dict.copy()


        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        elif from_flax:
            try:
                from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see "
                    "https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions."
                )
                raise
        elif from_pt:

            if low_cpu_mem_usage:
                cls._load_pretrained_model_low_mem(model, loaded_state_dict_keys, resolved_archive_file)
            else:
                model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                    model,
                    state_dict,
                    resolved_archive_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    sharded_metadata=sharded_metadata,
                    _fast_init=_fast_init,
                )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model
