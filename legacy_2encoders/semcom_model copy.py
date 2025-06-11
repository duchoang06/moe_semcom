import torch
import numpy as np 
import torch.nn as nn
from torch.nn import functional as F
from wireless_utils import QAMModem, ComplexWirelessChannel, SimpleWirelessChannel

from base_models import BERTTextEncoder, TaskPrompt, TaskConditionedTransformer, MultiTaskHead, TransformerDecoderWithSNR, TransformerDecoder

from moe_models import MoETransformer, HetereoMoETransformer

#considered task: sentiment analysis

class Transformer_SemCom(nn.Module):
    def __init__(self, num_tasks, embed_dim, task_dim, num_encd_layer, transmit_dim, rician_k=4.0):
        super().__init__()
        self.text_encoder = BERTTextEncoder(output_dim=embed_dim)
        self.task_prompt = TaskPrompt(num_tasks, task_dim)
        self.encoder_transformer = TaskConditionedTransformer(input_dim=embed_dim + task_dim, ffn_dim= (embed_dim + task_dim) * 4, num_layers=num_encd_layer)
        # self.decoder_transformer = TransformerDecoderWithSNR(input_dim=embed_dim + task_dim)
        self.decoder_transformer = TaskConditionedTransformer(input_dim=embed_dim + task_dim, ffn_dim= (embed_dim + task_dim) * 4, num_layers=num_encd_layer) 

        self.output_head = MultiTaskHead(embed_dim + task_dim, self.text_encoder.vocab_size, num_tasks)

        self.channel_encoder = nn.Sequential(
            nn.Linear(embed_dim + task_dim, int((embed_dim + task_dim)*4)),
            nn.ReLU(),
            nn.Linear(int((embed_dim + task_dim)*4), transmit_dim)
        )

        self.channel_decoder = nn.Sequential(
            nn.Linear(transmit_dim, int((embed_dim + task_dim)*4)),
            nn.ReLU(),
            nn.Linear(int((embed_dim + task_dim)*4), embed_dim + task_dim)
        )

        # self.physical_channel = SimpleWirelessChannel(snr_dB=15, fading='none')
        self.physical_channel = ComplexWirelessChannel(snr_dB=15, fading='none', rician_k=rician_k)

    def forward(self, text_list, task_id, snr, fading, rician_k=4.0):
        text_feat, input_ids, attn_mask = self.text_encoder(text_list)

        input_lengths = attn_mask.sum(dim=1) # text_feat: (64, seq_len, 768)

        task_feat = self.task_prompt(task_id, len(text_list)) # task_feat: (64, 16)
        seq_len = text_feat.size(1)
        task_feat = task_feat.unsqueeze(1).repeat(1, seq_len, 1) # task_feat: (64, seq_len, 16)

        fused = torch.cat([text_feat, task_feat], dim=-1) # fused: (batch, seq_len, embed_dim + task_dim): (64, seq_len, 784)

        # semantic encoded features
        semantic_encoded = self.encoder_transformer(fused)

        channel_encoded = self.channel_encoder(semantic_encoded) # (64, seq_len, 784) -> (64, seq_len, transmit_dim)

        rx_signal = self.physical_channel(channel_encoded, snr, fading, rician_k=rician_k) # (64, seq_len, 784)

        channel_decoded = self.channel_decoder(rx_signal) # (64, seq_len, transmit_dim) -> (64, seq_len, 784)
        
        # semantic decoded features
        semantic_decoded = self.decoder_transformer(channel_decoded) # (64, seq_len, 784)

        output = self.output_head(semantic_decoded, task_id, input_lengths, input_ids)

        return output, input_ids, input_lengths, semantic_encoded, semantic_decoded


class MoE_SemCom(nn.Module):
    def __init__(self, num_tasks, embed_dim, task_dim, transmit_dim, num_encd_layer, num_experts=4, rician_k=4.0):
        super().__init__()
        self.text_encoder = BERTTextEncoder(output_dim=embed_dim)
        self.task_prompt = TaskPrompt(num_tasks, task_dim)

        dim_feedforward = (embed_dim + task_dim) * 4
        top_k = 2

        self.encoder_transformer = MoETransformer(
            input_dim=embed_dim + task_dim,
            nhead=4,
            num_layers=num_encd_layer,
            num_experts=num_experts,
            top_k=top_k,
            dim_feedforward=dim_feedforward,
        )

        self.decoder_transformer = MoETransformer(
            input_dim=embed_dim + task_dim,
            nhead=4,
            num_layers=num_encd_layer,
            num_experts=num_experts,
            top_k=top_k,
            dim_feedforward=dim_feedforward,
        )

        self.output_head = MultiTaskHead(embed_dim + task_dim, self.text_encoder.vocab_size, num_tasks)

        # self.physical_channel = SimpleWirelessChannel(snr_dB=15, fading='none')
        self.physical_channel = ComplexWirelessChannel(snr_dB=15, fading='none', rician_k=rician_k)

        self.channel_encoder = nn.Sequential(
            nn.Linear(embed_dim + task_dim, int((embed_dim + task_dim)*4)),
            nn.ReLU(),
            nn.Linear(int((embed_dim + task_dim)*4), transmit_dim)
        )

        self.channel_decoder = nn.Sequential(
            nn.Linear(transmit_dim, int((embed_dim + task_dim)*4)),
            nn.ReLU(),
            nn.Linear(int((embed_dim + task_dim)*4), embed_dim + task_dim)
        )
        self.expert_sizes = self.encoder_transformer.expert_sizes



    def forward(self, text_list, task_id, snr, fading, rician_k=4.0):
        text_feat, input_ids, attn_mask = self.text_encoder(text_list)

        input_lengths = attn_mask.sum(dim=1) # text_feat: (64, seq_len, 768)

        task_feat = self.task_prompt(task_id, len(text_list))
        seq_len = text_feat.size(1)
        task_feat = task_feat.unsqueeze(1).repeat(1, seq_len, 1) # task_feat: (64, seq_len, 16)

        fused = torch.cat([text_feat, task_feat], dim=-1) # fused: (batch, seq_len, embed_dim + task_dim): (64, seq_len, 784)

        # semantic encoded features
        semantic_encoded, encoder_gate_scores, encoder_expert_masks = self.encoder_transformer(fused, snr) # (64, seq_len, 784)

        channel_encoded = self.channel_encoder(semantic_encoded) # (64, seq_len, 784) -> (64, seq_len, transmit_dim)

        rx_signal = self.physical_channel(channel_encoded, snr, fading, rician_k=rician_k) # (64, seq_len, 784)

        channel_decoded = self.channel_decoder(rx_signal) # (64, seq_len, transmit_dim) -> (64, seq_len, 784)

        # semantic decoded features
        semantic_decoded, decoder_gate_scores, decoder_expert_masks  = self.decoder_transformer(channel_decoded, snr) # (64, seq_len, 784)

        output = self.output_head(semantic_decoded, task_id, input_lengths, input_ids)

        return output, input_ids, input_lengths, semantic_encoded, semantic_decoded, encoder_gate_scores + decoder_gate_scores,  encoder_expert_masks + decoder_expert_masks


class HetereoMoE_SemCom(nn.Module):
    def __init__(self, num_tasks, embed_dim, task_dim, transmit_dim, num_encd_layer, num_experts=4, size_distribution='arithmetic', rician_k=4.0):
        super().__init__()
        self.text_encoder = BERTTextEncoder(output_dim=embed_dim)
        self.task_prompt = TaskPrompt(num_tasks, task_dim)

        dim_feedforward = (embed_dim + task_dim) * 4
        top_k = 2

        self.encoder_transformer = HetereoMoETransformer(
            input_dim=embed_dim + task_dim,
            nhead=4,
            num_layers=num_encd_layer,
            num_experts=num_experts,
            top_k=top_k,
            dim_feedforward=dim_feedforward,
            size_distribution=size_distribution,
        )

        self.decoder_transformer = HetereoMoETransformer(
            input_dim=embed_dim + task_dim,
            nhead=4,
            num_layers=num_encd_layer,
            num_experts=num_experts,
            top_k=top_k,
            dim_feedforward=dim_feedforward,
            size_distribution=size_distribution,
        )


        self.channel_encoder = nn.Sequential(
            nn.Linear(embed_dim + task_dim, int((embed_dim + task_dim)*4)),
            nn.ReLU(),
            nn.Linear(int((embed_dim + task_dim)*4), transmit_dim)
        )

        self.channel_decoder = nn.Sequential(
            nn.Linear(transmit_dim, int((embed_dim + task_dim)*4)),
            nn.ReLU(),
            nn.Linear(int((embed_dim + task_dim)*4), embed_dim + task_dim)
        )


        self.output_head = MultiTaskHead(embed_dim + task_dim, self.text_encoder.vocab_size, num_tasks)

        # self.physical_channel = SimpleWirelessChannel(snr_dB=15, fading='none')
        self.physical_channel = ComplexWirelessChannel(snr_dB=15, fading='none', rician_k=rician_k)
        self.expert_sizes = self.encoder_transformer.expert_sizes


    def forward(self, text_list, task_id, snr, fading, rician_k=4.0):
        text_feat, input_ids, attn_mask = self.text_encoder(text_list)

        input_lengths = attn_mask.sum(dim=1) # text_feat: (64, seq_len, 768)

        task_feat = self.task_prompt(task_id, len(text_list))
        seq_len = text_feat.size(1)
        task_feat = task_feat.unsqueeze(1).repeat(1, seq_len, 1) # task_feat: (64, seq_len, 16)

        fused = torch.cat([text_feat, task_feat], dim=-1) # fused: (batch, seq_len, embed_dim + task_dim): (64, seq_len, 784)

        # semantic encoded features
        semantic_encoded, encoder_gate_scores, encoder_expert_masks = self.encoder_transformer(fused, snr) # (64, seq_len, 784)

        channel_encoded = self.channel_encoder(semantic_encoded) # (64, seq_len, 784) -> (64, seq_len, transmit_dim)

        rx_signal = self.physical_channel(channel_encoded, snr, fading, rician_k=rician_k) # (64, seq_len, 784)

        channel_decoded = self.channel_decoder(rx_signal) # (64, seq_len, transmit_dim) -> (64, seq_len, 784)

        # semantic decoded features
        semantic_decoded, decoder_gate_scores, decoder_expert_masks  = self.decoder_transformer(channel_decoded, snr) # (64, seq_len, 784)

        output = self.output_head(semantic_decoded, task_id, input_lengths, input_ids)

        return output, input_ids, input_lengths, semantic_encoded, semantic_decoded, encoder_gate_scores + decoder_gate_scores,  encoder_expert_masks + decoder_expert_masks


        

        
