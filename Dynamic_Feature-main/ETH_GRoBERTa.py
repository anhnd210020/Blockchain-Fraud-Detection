import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# Correct imports for transformers library
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaModel,
    RobertaPooler,
)


class VocabGraphConvolution(nn.Module):
    def __init__(self, voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.voc_dim = voc_dim
        self.num_adj = num_adj
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        for i in range(self.num_adj):
            setattr(
                self, "W%d_vh" % i, nn.Parameter(torch.randn(voc_dim, hid_dim))
            )

        self.fc_hc = nn.Linear(hid_dim, out_dim)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if (
                    n.startswith("W")
                    or n.startswith("a")
                    or n in ("W", "a", "dense")
            ):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            # H_vh = vocab_adj_list[i].mm(getattr(self, "W%d_vh" % i))
            if not isinstance(vocab_adj_list[i], torch.Tensor) or not vocab_adj_list[i].is_sparse:
                raise TypeError("Expected a PyTorch sparse tensor")
            H_vh = torch.sparse.mm(vocab_adj_list[i].float(), getattr(self, "W%d_vh" % i))

            # H_vh=self.dropout(F.elu(H_vh))
            H_vh = self.dropout(H_vh)
            H_dh = X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear = X_dv.matmul(getattr(self, "W%d_vh" % i))
                H_linear = self.dropout(H_linear)
                H_dh += H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out = self.fc_hc(fused_H)
        return out


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    """
    实现 DiffSoftmax，用于在训练中使用软标签或硬标签。
    - tau: 温度参数，控制 softmax 输出的平滑度
    - hard: 是否使用硬标签
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class DynamicFusionLayer(nn.Module):
    def __init__(self, hidden_dim, tau=1.0, hard_gate=False):
        super(DynamicFusionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 3),
            # nn.Softmax(dim=-1),
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_embeddings, gcn_enhanced_embeddings):
        concat_embeddings = torch.cat([bert_embeddings, gcn_enhanced_embeddings], dim=-1)

        gate_logits = self.gate_network(concat_embeddings)
        gate_values = DiffSoftmax(gate_logits, tau=self.tau, hard=self.hard_gate, dim=-1)

        gate_bert_only = gate_values[:, :, 0].unsqueeze(-1)
        gate_gcn_enhanced = gate_values[:, :, 1].unsqueeze(-1)
        gate_gcn_bert_weighted = gate_values[:, :, 2].unsqueeze(-1)

        embeddings_bert_only = bert_embeddings
        embeddings_gcn_enhanced = gcn_enhanced_embeddings
        embeddings_gcn_bert_weighted = self.fusion_weight * bert_embeddings + (1 - self.fusion_weight) * gcn_enhanced_embeddings

        fused_embeddings = (
                gate_bert_only * embeddings_bert_only +
                gate_gcn_enhanced * embeddings_gcn_enhanced +
                gate_gcn_bert_weighted * embeddings_gcn_bert_weighted
        )

        return fused_embeddings


class ETH_GRoBERTaEmbeddings(RobertaEmbeddings):
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super(ETH_GRoBERTaEmbeddings, self).__init__(config)
        assert gcn_embedding_dim >= 0
        self.gcn_embedding_dim = gcn_embedding_dim
        self.vocab_gcn = VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim)

        self.dynamic_fusion_layer = DynamicFusionLayer(config.hidden_size)

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, attention_mask=None):
        words_embeddings = self.word_embeddings(input_ids)

        vocab_input = gcn_swop_eye.matmul(words_embeddings).transpose(1, 2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)

        gcn_words_embeddings = words_embeddings.clone()
        for i in range(self.gcn_embedding_dim):
            tmp_pos = (attention_mask.sum(-1) - 2 - self.gcn_embedding_dim + 1 + i
                       ) + torch.arange(0, input_ids.shape[0]).to(input_ids.device) * input_ids.shape[1]
            gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = gcn_vocab_out[:, :, i]

        new_words_embeddings = self.dynamic_fusion_layer(words_embeddings, gcn_words_embeddings)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = new_words_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ETH_GRoBERTaModel(RobertaModel):
    def __init__(
            self,
            config,
            gcn_adj_dim,
            gcn_adj_num,
            gcn_embedding_dim,
            num_labels,
            output_attentions=False,
            keep_multihead_output=False,
    ):
        super(ETH_GRoBERTaModel, self).__init__(config)
        self.embeddings = ETH_GRoBERTaEmbeddings(
            config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim
        )
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.output_attentions = config.output_attentions if hasattr(config, 'output_attentions') else False
        self.keep_multihead_output = config.keep_multihead_output if hasattr(config, 'keep_multihead_output') else False
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Extract custom kwargs (these are positional in __init__ but passed as kwargs)
        gcn_adj_dim = kwargs.pop('gcn_adj_dim', None)
        gcn_adj_num = kwargs.pop('gcn_adj_num', None)
        gcn_embedding_dim = kwargs.pop('gcn_embedding_dim', None)
        num_labels = kwargs.pop('num_labels', None)
        output_attentions = kwargs.pop('output_attentions', False)
        keep_multihead_output = kwargs.pop('keep_multihead_output', False)

        # Get config
        config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)

        # Create the model with custom params
        model = cls(
            config, 
            gcn_adj_dim, 
            gcn_adj_num, 
            gcn_embedding_dim, 
            num_labels, 
            output_attentions, 
            keep_multihead_output
        )

        # Load pretrained state_dict for base RoBERTa parts
        state_dict = RobertaModel.from_pretrained(pretrained_model_name_or_path, config=config).state_dict()
        model.load_state_dict(state_dict, strict=False)  # strict=False to ignore custom layers like GCN/fusion

        return model

    def forward(
            self,
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            attention_mask=None,
            output_all_encoded_layers=False,
            head_mask=None,
    ):
        vocab_adj_list = [adj * 0 for adj in vocab_adj_list]  # Note: This zeros out adj; consider removing if unintended

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            attention_mask,
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=self.dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=output_all_encoded_layers,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits