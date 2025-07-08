import numpy as np 
import datasets
from utils import QQPPromptDataset
from torch.utils.data import DataLoader
from wireless_utils import QAMModem, ComplexWirelessChannel, SimpleWirelessChannel
import torch 
import torch.nn as nn
import torch.nn.functional as F

from pyldpc import make_ldpc, decode, get_message, encode
import pyldpc
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from base_models import BERTTextEncoder
from multiprocessing import Pool
import os
from joblib import Parallel, delayed

def _get_message_wrapper(args):
    G, i = args
    return get_message(G, i)

def _decode_and_get_message(args):
    H, G, i, snr, max_iter = args
    decoded = decode(H, i, snr=snr, maxiter=max_iter)
    return get_message(G, decoded)

class UTF8SourceEncoder(nn.Module):
    def __init__(self, max_bits=2400):
        super().__init__()
        self.max_bits = max_bits

    def forward(self, x: list[str]) -> torch.Tensor:
        encoded = [s.encode('utf-8')[:self.max_bits // 8] for s in x] # encoding to utf-8 bytes, truncating to max_bits
        bit_arrays = []
        for e in encoded:
            bits = ''.join(f'{byte:08b}' for byte in e)
            padded_bits = bits.ljust(self.max_bits, '0')
            bit_array = [int(b) for b in padded_bits]
            bit_arrays.append(bit_array)
        return torch.tensor(bit_arrays, dtype=torch.float32)

class UTF8SourceDecoder(nn.Module):
    def __init__(self, max_bits=2400):
        super().__init__()
        self.max_bits = max_bits

    def forward(self, x: torch.Tensor) -> list[str]:
        x = x.round().int()  # Ensure values are 0 or 1
        decoded_strings = []
        for bits in x:
            # print("Decoded bits (first 32):", bits[:32])
            # print("Sum of bits:", bits.sum().item())

            # Only decode full bytes
            full_bytes = self.max_bits // 8
            bit_str = ''.join(str(b.item()) for b in bits[:full_bytes * 8])

            # Convert to byte list
            byte_list = [int(bit_str[i:i+8], 2) for i in range(0, len(bit_str), 8)]

            try:
                decoded = bytes(byte_list).decode('utf-8', errors='ignore')
            except:
                decoded = ''  # fallback in case of decoding error
            decoded_strings.append(decoded)
        return decoded_strings


class LDPCEncoder(nn.Module):
    def __init__(self, n_bits=2400, d_v=2, d_c=4):
        super().__init__()
        # Construct LDPC H and G matrices
        self.H, self.G = make_ldpc(n_bits, d_v, d_c, systematic=True, sparse=True)
        # G = G.astype(np.uint8)
        # self.register_buffer('G', torch.from_numpy(G).float())
        self.k = self.G.shape[1]

    def forward(self, m, snr):
        encoded_arr = []

        for msg in m:
            # msg_encoded = encode(self.G, msg.cpu(), snr=snr)
            # n, k = self.G.shape
            d = pyldpc.utils.binaryproduct(self.G, msg) 
            msg_encoded = (-1) ** d

            encoded_arr.append(msg_encoded)

        encoded_arr = np.stack(encoded_arr, axis=0)
        return encoded_arr
    
    
class LDPCDecoder(nn.Module):
    def __init__(self, H, G, snr_dB=12.0, max_iter=1e2):
        super().__init__()
        self.H = H
        self.G = G
        self.snr_dB   = snr_dB
        self.max_iter = int(max_iter)
        self.n, self.k = G.shape

    def forward(self, y, snr=20.0):
        # decoded_arr = []
        # for i in y:
        #     decoded = decode(self.H, i, snr=snr, maxiter=self.max_iter)
        #     x = get_message(self.G, decoded)
        #     decoded_arr.append(x)

        args_list = [(self.H, self.G, i, snr, self.max_iter) for i in y]

        # with Pool(processes=os.cpu_count()) as pool:
            # decoded_arr = pool.map(_decode_and_get_message, args_list)

        decoded_arr = Parallel(n_jobs=32 - 1, backend='loky')(
            delayed(_decode_and_get_message)(args) for args in args_list
        )

        return np.stack(decoded_arr, axis=0)

        # tY = y.reshape(-1, len(y)) 
        # decoded = decode(self.H, tY, snr=snr, maxiter=self.max_iter)

        # # decoded_arr = []
        # decoded = decoded.reshape(len(y), -1)  

        # # for i in decoded:
        # #     x = get_message(self.G, i)
        # #     decoded_arr.append(x)

        # with Pool() as pool:
        #     decoded_arr = pool.map(_get_message_wrapper, [(self.G, i) for i in decoded])

        # print(f'decoding done, time {np.datetime64("now")}')

        # return np.stack(decoded_arr, axis=0)

class Conventional_Com(nn.Module):
    def __init__(self, max_char_len=600, emded_dim=512):
        super().__init__()
        self.max_bits_codeword = max_char_len * 8

        # Modules
        self.channel_encoder = LDPCEncoder(self.max_bits_codeword, d_v=2, d_c=4)

        self.channel_decoder = LDPCDecoder(self.channel_encoder.H, self.channel_encoder.G, snr_dB=12.0, max_iter=1e2)

        self.max_bits_msg = self.channel_encoder.k

        self.source_encoder   = UTF8SourceEncoder(max_bits=self.max_bits_msg)

        self.physical_channel = SimpleWirelessChannel(snr_dB=12.0, fading='none', rician_k=3.0) 

        self.source_decoder   = UTF8SourceDecoder(max_bits=self.max_bits_msg)

        self.text_encoder = BERTTextEncoder(output_dim=emded_dim, max_seq_len=64)

        self.classifier_head = nn.Linear(emded_dim, 2) 

        # self.classifer = RNNClassifier(
        #     vocab_size=self.text_encoder.vocab_size,
        #     emb_dim=emded_dim,
        #     hidden_dim=emded_dim*2,
        #     n_classes=2,
        #     pad_idx=self.text_encoder.tokenizer.pad_token_id
        # # 
        
    def forward(self, x, snr=12.0, fading='none', rician_k=3.0):
        device = next(self.parameters()).device

        source_encoded = self.source_encoder(x) 


        channel_encoded = self.channel_encoder(source_encoded, snr=40.0) # noiseless encoding (B, num_bits_codeword)

        channel_encoded = torch.tensor(channel_encoded, dtype=torch.float32)

        rx_signal = self.physical_channel(channel_encoded, snr=snr, fading=fading, rician_k=rician_k, modal=False)

        rx_signal = np.array(rx_signal, dtype=np.float64)

        channel_decoded = self.channel_decoder(rx_signal, snr=40.0)

        channel_decoded = torch.tensor(channel_decoded)  

        source_decoded = self.source_decoder(channel_decoded)

        rcvd_strings = [s[:len(_m)] for (s, _m)  in zip(source_decoded, x)]

        text_feat, input_ids, attn_mask = self.text_encoder(rcvd_strings)

        seq_repr = text_feat.mean(dim=1)  

        logits = self.classifier_head(seq_repr) 

        return logits


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, lengths):
        # input_ids: [B, Lmax]; lengths: [B]
        emb = self.emb(input_ids)  # [B, Lmax, emb_dim]
        # pack, run LSTM, unpack
        packed = pack_padded_sequence(emb, lengths.cpu(), 
                                      batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.rnn(packed)
        # hn[-1]: [B, hidden_dim]
        return self.fc(hn[-1])
