import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random, os
from torch.utils.data import Dataset, DataLoader


def text_loss(outputs, labels, task_id, input_ids=None, input_lengths=None):
    loss = 0.0
    if task_id == 0:
        logit = outputs
        label = labels
        loss = F.cross_entropy(logit, label)

    elif task_id == 1:
        target_len = outputs.size(1)

        pred_logits = outputs.transpose(1, 2)  # [batch, vocab_size, seq_len]
        target_tokens = input_ids[:, :target_len] 

        loss = F.cross_entropy(pred_logits, target_tokens, ignore_index=0)

    else:
        raise ValueError(f"Unknown task_id: {task_id}")
                
    return loss

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # x: (N, d), y: (N, d)
        inp = torch.cat([x, y], dim=-1)
        return self.net(inp)

def mutual_information_loss(tx_signal, rx_signal, critic):
    # Collapse batch and sequence dims for simplicity
    B, T, d = tx_signal.shape
    x = tx_signal.reshape(-1, d)
    y = rx_signal.reshape(-1, d)

    # Positive samples: joint distribution
    joint_score = critic(x, y)   # (B*T, 1)

    # Negative samples: shuffle y to break the correspondence
    y_perm = y[torch.randperm(y.size(0))]
    marg_score = critic(x, y_perm)  # (B*T, 1)

    # Donsker-Varadhan lower bound
    mi_estimate = joint_score.mean() - torch.log(torch.exp(marg_score).mean() + 1e-8)
    # For loss (minimize), use negative MI
    return -mi_estimate


def moe_balancing_loss_p_penalty(all_gate_scores, all_expert_masks, expert_sizes):
    '''
        parameter-penalty loss adapted from "HMoE: Heterogeneous Mixture of Experts for Language Modeling"
    '''
    # all_expert_mask: (num_layers, num_tokens, num_experts)
    loss = 0.0
    N = expert_sizes.shape[0]

    for gate_scores, expert_mask in zip(all_gate_scores, all_expert_masks):
        T, N_layer = gate_scores.shape
        assert N_layer == N, "Mismatch in number of experts"

        # P = torch.softmax(gate_scores, dim=-1)  # (T, N)
        P = gate_scores # already softmaxed in the model

        P_hat = P.mean(dim=0)  # (N,)
        M = (expert_mask.float() * expert_sizes.view(1, -1)).mean(dim=0)  # (N,)
        loss += N * torch.sum(M * P_hat)

    return loss #to-do: may return 0.0 here
    # return torch.tensor(0.0)

def moe_balancing_loss(all_gate_scores, all_expert_masks, expert_sizes):
    '''
    Balancing loss for HMoE: encourages routing proportional to expert sizes.
    '''
    loss = 0.0
    ideal_load = expert_sizes / expert_sizes.sum()  # (N,)

    for gate_scores in all_gate_scores:  # gate_scores: (T, N), softmaxed
        actual_load = gate_scores.mean(dim=0)  # (N,)
        # Small constant for numerical stability
        actual_load = actual_load + 1e-8
        loss += torch.nn.functional.kl_div(actual_load.log(), ideal_load, reduction='batchmean')

    return loss



class SST2Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["sentence"]
        label = item["label"]
        return text, label
    
def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels)

def get_test_loader_for_epoch(epoch, validation_data, seed, num_samples=5):
    # Use epoch-dependent seed
    random.seed(seed + epoch)
    sampled_data = random.sample(list(validation_data), k=num_samples)
    
    test_dataset = SST2Dataset(sampled_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return test_loader


def snr_loss(semantic_decoded, semantic_encoded, snr_tensor):
    """
    Encourages the decoded semantic features to remain close to the
    original (pre-channel) semantic representation, especially at high SNR.
    Args:
        semantic_decoded: [batch, dim] - output of the decoder
        semantic_encoded: [batch, dim] - output before the channel (clean)
        snr_tensor:       [batch, 1]   - SNR in dB for each sample

    Returns:
        scalar loss (mean over batch)
    """
    with torch.no_grad():
        clean_feat = semantic_encoded.detach()

    snr_linear = 10 ** (snr_tensor / 10)  # convert dB to linear scale
    weight = 1 / (snr_linear + 1e-6)      # lower SNR -> weaker penalty

    loss = ((semantic_decoded - clean_feat) ** 2).mean(dim=1) * weight.squeeze()
    return loss.mean()



def fix_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def sample_batch(data, batch_size=8):
    batch = random.sample(list(data), batch_size)
    texts = [sample['sentence'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch])
    return texts, labels

def sample_mixed_task_batch(data, batch_size=8):
    batch = random.sample(list(data), batch_size)

    batch_dicts = []
    for i, sample in enumerate(batch):
        task_id = 0 if i < batch_size // 2 else 1  # First half: classification, second: reconstruction
        batch_dicts.append({
            'text': sample['sentence'],
            'label': sample['label'],   # Only used for classification
            'task_id': task_id
        })

    # Shuffle to mix task types
    random.shuffle(batch_dicts)

    # Convert to lists/tensors
    texts = [item['text'] for item in batch_dicts]
    task_ids = torch.tensor([item['task_id'] for item in batch_dicts])
    labels = torch.tensor([item['label'] for item in batch_dicts])

    return texts, labels, task_ids

def sample_single_task_batch(task_id, data, batch_size=8):
    batch = random.sample(list(data), batch_size)

    batch_dicts = []
    for sample in batch:
        batch_dicts.append({
            'text': sample['sentence'],
            'label': sample['label'],  # Used only for classification
            'task_id': task_id
        })

    texts = [item['text'] for item in batch_dicts]
    task_ids = torch.tensor([item['task_id'] for item in batch_dicts])
    labels = torch.tensor([item['label'] for item in batch_dicts])

    return texts, labels, task_ids


def estimate_transformer_flops(num_blocks, layers_per_block, d_model, d_ff, seq_len, batch_size= 1):
    # Attention FLOPs per layer (Q, K, V, attn score, softmax, context proj)
    flops_attention = 4 * seq_len * d_model**2 + 2 * seq_len**2 * d_model

    # Feedforward FLOPs per layer
    flops_ffn = 4 * seq_len * d_model * d_ff

    # FLOPs per transformer encoder layer
    flops_per_layer = flops_attention + flops_ffn

    # Total layers
    total_layers = num_blocks * layers_per_block

    # Multiply by total layers and batch
    total_flops = flops_per_layer * total_layers * batch_size

    return total_flops

def estimate_moe_flops(num_blocks, layers_per_block, d_model, d_ff, seq_len, batch_size, num_experts):
    pass 

