import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AlbertModel, AlbertPreTrainedModel

class AlbertForTextRanking(AlbertPreTrainedModel):
    R"""
    The text ranking model using albert with siamese network.
    """
    def __init__(self, congfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.query_albert = AlbertModel(config)
        self.passage_albert = AlbertModel(config)
        self.query_project = nn.Linear(config.hidden_size, config.proj_size)
        self.passage_project = nn.Linear(config.hidden_size, config.proj_size)

        project_dropout = (
                config.project_dropout_prob if config.project_dropout_prob is not None \
                        else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(project_dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weight()

    def forward(self, 
                query_input_ids=None, passage_input_ids=None, 
                query_attention_mask=None, passage_attention_mask=None,
                query_token_type_ids=None, passage_token_type_ids=None
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

        outputs = self.query_albert(
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

        outputs = self.passage_albert(
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

        with torch.no_grad():
            outputs = self.forward(**inputs)
            probabilities = self.softmax(self.tokens_clf(outputs.hidden_states[-1]))
            predictions = torch.argmax(probabilities, dim=-1)

            # active filtering
            active_tokens = inputs['attention_mask'] == 1
            active_predictions = torch.where(
                active_tokens,
                predictions,
                torch.tensor(-1).type_as(predictions)
            )
            return {"probabilities": probabilities[:, :, 1].detach(), # shape: (batch, length)
                    "active_predictions": predictions.detach(),
                    "active_tokens": active_tokens}


