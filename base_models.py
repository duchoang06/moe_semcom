import torch
import torch.nn as nn
import numpy as np 
from transformers import BertModel, BertTokenizer


class BERTTextEncoder(nn.Module):
    def __init__(self, output_dim=768, bert_model_name='bert-base-uncased'):
        super(BERTTextEncoder, self).__init__()
        
        # Load pretrained BERT model and tokenizer
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name) # detault embedding size is 768
        
        self.projection = nn.Linear(self.bert_model.config.hidden_size, output_dim) # 768 to output_dim=256
        self.vocab_size = self.tokenizer.vocab_size

    def forward(self, text_list):
        device = next(self.parameters()).device
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=64).to(device)

        with torch.no_grad():
            outputs = self.bert_model(**encoded_input)
        
        # cls_embedding = outputs.last_hidden_state[:, 0, :]
        text_embedding = outputs.last_hidden_state  

        text_embedding = self.projection(text_embedding)  # Shape: (batch_size, embed_dim) 

        return text_embedding, encoded_input['input_ids'], encoded_input['attention_mask'] # encoded_input['input_ids'] is for reconstruction 


class TaskPrompt(nn.Module):
    def __init__(self, num_tasks, task_dim):
        super(TaskPrompt, self).__init__()
        self.task_embeddings = nn.Embedding(num_tasks, task_dim)

    def forward(self, task_id, batch_size):
        device = self.task_embeddings.weight.device  # get device of the embedding
        task_ids = torch.full((batch_size,), task_id, dtype=torch.long, device=device)
        return self.task_embeddings(task_ids)
    
class TaskConditionedTransformer(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2, ffn_dim=2048):
        super(TaskConditionedTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True, dim_feedforward=ffn_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer(x)
        return x
    
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, vocab_size, num_tasks=2):
        super().__init__()
        # self.heads = nn.ModuleDict({
        #     'txt_cls': nn.Sequential(
        #         nn.Linear(input_dim, input_dim//2),
        #         nn.ReLU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(input_dim//2, 2)
        #     ),
        #     'txt_res': TransformerTaskHead(input_dim, vocab_size) 
        # })
        self.heads = nn.ModuleDict({
            'txt_cls': nn.Linear(input_dim, 2),
            'txt_res': nn.Linear(input_dim, vocab_size), 
        })

    def forward(self, x, task_id, input_lengths=None, input_ids=None):
        if task_id == 1:  # sequence reconstruction
            head_output = self.heads['txt_res'](x)
        elif task_id == 0:  # classification
            seq_repr = x.mean(dim=1) #to-do: consider using cls token
            head_output = self.heads['txt_cls'](seq_repr)

        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        return head_output


class SemanticDecoder(nn.Module):
    def __init__(self, input_dim, vocab_size, ffn_dim=2024, max_seq_len=64, nhead=4, num_layers=4, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, input_dim))
        # self.vocab_proj = nn.Linear(input_dim, vocab_size)
        self.pad_token_id = pad_token_id

        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=ffn_dim, batch_first=True)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory, input_lengths=None, memory_key_padding_mask=None):
        """
        Args:
            channel_out: [batch, seq_len, input_dim] — output from channel or encoder, to be reconstructed.
            input_lengths: (optional) mask for variable lengths
            memory_key_padding_mask: [batch, seq_len] bool mask (optional)
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = memory.size(1)
        x = memory + self.pos_embed[:seq_len].unsqueeze(0)  # [B, L, D]

        # No tgt_mask (no causal mask!), tgt == memory == x
        out = self.transformer_decoder(
            tgt=x,                # [B, L, D]
            memory=x,             # [B, L, D]
            tgt_mask=None,
            tgt_key_padding_mask=None,         # Only needed if you want to mask padding in tgt (here not needed)
            memory_key_padding_mask=memory_key_padding_mask
        )
        # logits = self.vocab_proj(out)  # [B, L, vocab_size]
        return out


    
class TransformerTaskHead_Autoregressive(nn.Module):
    def __init__(self, input_dim, vocab_size, max_seq_len=64, nhead=4, num_layers=2, pad_token_id=0, sos_token_id=101):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id

        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, input_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.vocab_proj = nn.Linear(input_dim, vocab_size)

    def forward(self, memory, seq_len, input_ids=None, memory_key_padding_mask=None):
        device = memory.device
        batch_size = memory.size(0)

        if self.training:
            assert input_ids is not None, "input_ids must be provided during training"
            # each input_id is fully [101, ... , 102, 0,..] (with paddings)

            tgt_input_ids = input_ids[:, :-1]  # shift for teacher forcing
            tgt_seq_len = tgt_input_ids.size(1)

            # Cap length to match positional embeddings
            max_len = self.pos_embed.size(0)
            if tgt_seq_len > max_len:
                tgt_input_ids = tgt_input_ids[:, :max_len]
                tgt_seq_len = max_len

            tgt_embed = self.embedding(tgt_input_ids)
            tgt_embed = tgt_embed + self.pos_embed[:tgt_seq_len].unsqueeze(0)

            tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()
            tgt_key_padding_mask = (tgt_input_ids == self.pad_token_id)

            out = self.transformer_decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

            logits = self.vocab_proj(out)  # [B, tgt_seq_len, vocab_size]
            return logits

        else:
            # Inference mode: greedy decoding
            generated = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            outputs = []

            for _ in range(memory.size(1)):
                tgt_embed = self.embedding(generated)
                tgt_embed = tgt_embed + self.pos_embed[:generated.size(1)].unsqueeze(0)

                tgt_mask = torch.triu(torch.ones(generated.size(1), generated.size(1), device=device), diagonal=1).bool()

                decoder_out = self.transformer_decoder(
                    tgt=tgt_embed,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )

                logits = self.vocab_proj(decoder_out[:, -1, :])  # [B, vocab]
                next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]

                finished = finished | (next_token.squeeze() == self.pad_token_id)
                # finished = finished | (next_token.squeeze() == 103) # fake token id

                next_token[finished] = self.pad_token_id

                generated = torch.cat([generated, next_token], dim=1)
                outputs.append(next_token)

                if finished.all():
                    break

            generated = generated[:, 1:]
            # print(f"Generated sequence: {generated}")

            return generated  # remove <sos>

