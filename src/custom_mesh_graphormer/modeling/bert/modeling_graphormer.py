"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import code
import torch
from torch import nn
from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertPooler, BertIntermediate, BertOutput, BertSelfOutput
import custom_mesh_graphormer.modeling.data.config as cfg
from custom_mesh_graphormer.modeling._gcnn import GraphConvolution, GraphResBlock
from .modeling_utils import prune_linear_layer
LayerNormClass = torch.nn.LayerNorm
BertLayerNorm = torch.nn.LayerNorm
from comfy.model_management import get_torch_device
device = get_torch_device()


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask,
                history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class GraphormerLayer(nn.Module):
    def __init__(self, config):
        super(GraphormerLayer, self).__init__()
        self.attention = BertAttention(config)
        self.has_graph_conv = config.graph_conv
        self.mesh_type = config.mesh_type

        if self.has_graph_conv == True:
            self.graph_conv = GraphResBlock(config.hidden_size, config.hidden_size, mesh_type=self.mesh_type)
        
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def MHA_GCN(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)
        attention_output = attention_outputs[0]

        if self.has_graph_conv==True:
            if self.mesh_type == 'body':
                joints = attention_output[:,0:14,:] 
                vertices = attention_output[:,14:-49,:]
                img_tokens = attention_output[:,-49:,:]

            elif self.mesh_type == 'hand':
                joints = attention_output[:,0:21,:]
                vertices = attention_output[:,21:-49,:]
                img_tokens = attention_output[:,-49:,:]

            vertices = self.graph_conv(vertices)
            joints_vertices = torch.cat([joints,vertices,img_tokens],dim=1)
        else:
            joints_vertices = attention_output

        intermediate_output = self.intermediate(joints_vertices)
        layer_output = self.output(intermediate_output, joints_vertices)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        return self.MHA_GCN(hidden_states, attention_mask, head_mask,history_state)


class GraphormerEncoder(nn.Module):
    def __init__(self, config):
        super(GraphormerEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([GraphormerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # outputs, (hidden states), (attentions)

class EncoderBlock(BertPreTrainedModel):
    def __init__(self, config):
        super(EncoderBlock, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = GraphormerEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim 

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).to(device)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        encoder_outputs = self.encoder(embeddings,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs

class Graphormer(BertPreTrainedModel):
    '''
    The archtecture of a transformer encoder block we used in Graphormer
    '''
    def __init__(self, config):
        super(Graphormer, self).__init__(config)
        self.config = config
        self.bert = EncoderBlock(config)
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None):
        '''
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        '''
        predictions = self.bert(img_feats=img_feats, input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
        pred_score = self.cls_head(predictions[0])
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats

        if self.config.output_attentions and self.config.output_hidden_states:
            return pred_score, predictions[1], predictions[-1]
        else:
            return pred_score

 