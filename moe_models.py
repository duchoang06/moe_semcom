import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    MoE system: output = \sum_i SNRAwareGating_i(x) * expert_i(x), where x in R^d is the input token.
'''

class SNRAwareGating(nn.Module):
    '''
    Input:
        - x: input features of shape (B, L, D)
        - snr: SNR values of shape (B, L)
    Output:
        - y: expert selection probabilities of shape (B*L, num_experts)     
    '''
    def __init__(self, input_dim, num_experts, tau=1.0, hard=False):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim + 1, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_experts)
        )
        self.tau = tau
        self.hard = hard

    def gumbel_softmax(self, logits):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        y = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)

        if self.hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y  # Straight-through estimator
        return y

    def forward(self, x, snr):
        B, L, D = x.shape
        x_flat = x.view(B * L, D)

        # Ensure snr is a tensor
        # if isinstance(snr, (int, float)):
        snr = torch.tensor(snr, dtype=torch.float32, device=x.device).expand(B, 1)

        snr = snr.view(B, 1).repeat(1, L).view(B * L, 1)  # expand to match token count
        gate_input = torch.cat([x_flat, snr], dim=-1)  # (B*L, D+1)
        logits = self.gate(gate_input)  # (B*L, num_experts)
        return self.gumbel_softmax(logits)  # (B*L, num_experts)

class LinearGating(nn.Module):
    def __init__(self, input_dim, num_experts, tau=1.0, hard=False):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim + 1, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_experts)
        )
        self.tau = tau
        self.hard = hard

    def gumbel_softmax(self, logits):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        y = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)

        if self.hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y  # Straight-through estimator
        return y

    def forward(self, x, snr):
        B, L, D = x.shape
        x_flat = x.view(B * L, D)

        logits = self.gate(x_flat)  # (B*L, num_experts)
        return self.gumbel_softmax(logits)  # (B*L, num_experts)

# ----------  MoE Transformer ----------
class ExpertFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2, tau=1.0, hard=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        self.gate = LinearGating(input_dim=input_dim, num_experts=num_experts, tau=tau, hard=hard)

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(B * L, D)
        gate_scores = self.gate(x)  # (B*L, num_experts)
        T = B * L

        # Compute the softmax probabilities by the gating, based on that select the top-k experts.
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (B*L, top_k)

        output = torch.zeros_like(x_flat)
        expert_mask = torch.zeros((T, self.num_experts), device=x.device)  # (T, N)


        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # (B*L,)
            weight = topk_scores[:, k].unsqueeze(1)  # (B*L, 1)
            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.sum() == 0:
                    continue
                x_selected = x_flat[mask]  # (N_i, D)
                out = self.experts[i](x_selected)  # (N_i, D)
                output[mask] += out * weight[mask]

                expert_mask[mask, i] = 1

        return output.view(B, L, D), gate_scores, expert_mask
    

class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, num_experts=4, top_k=2, tau=1.0, hard=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.moe_ffn = ExpertFFN(
            input_dim=d_model,
            hidden_dim=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            tau=tau,
            hard=hard
        )

    def forward(self, x):
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x2)

        x2, gate_scores, expert_mask = self.moe_ffn(x)
        x = self.norm2(x + x2)

        return x, gate_scores, expert_mask
    

class MoETransformer(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2, num_experts=4, top_k=2, dim_feedforward=2048, tau=1.0, hard=False):
        super().__init__()
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_experts=num_experts,
                top_k=top_k,
                tau=tau,
                hard=hard
            ) for _ in range(num_layers)
        ])

        self.expert_sizes = torch.zeros(num_experts, dtype=torch.int32)
        self.expert_sizes.fill_(dim_feedforward)


    def forward(self, x):
        all_gate_scores = []
        all_expert_masks = []

        for layer in self.layers:
            x, gate_scores, expert_mask = layer(x)
            all_gate_scores.append(gate_scores)
            all_expert_masks.append(expert_mask)
        return x, all_gate_scores, all_expert_masks



# ---------- Heterogeneous MoE Transformer ----------
class HetereoExpertFFN(nn.Module):
    '''
        Design: each expert has its own hid_dim, size variation following the arithmetic, geometric series  
    '''
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2, tau=1.0, hard=False, size_distribution='uniform'):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.size_distribution = size_distribution

        self.expert_sizes = torch.zeros(num_experts, dtype=torch.int32)

        if size_distribution == 'uniform':
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                ) for _ in range(num_experts)
            ])
            self.expert_sizes.fill_(hidden_dim)
            
        elif size_distribution == 'arithmetic':
            gap = int(hidden_dim * 0.125)
            a = hidden_dim - ((num_experts - 1) * gap) // 2

            # Ensure all expert sizes are >= 1
            self.expert_sizes = torch.tensor(
                [max(1, a + i * gap) for i in range(num_experts)],
                dtype=torch.int32
            )

            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.ReLU(),
                    nn.Linear(expert_dim, input_dim)
                ) for expert_dim in self.expert_sizes
            ])

        # elif size_distribution == 'arithmetic':
            # self.experts = nn.ModuleList([
            #     nn.Sequential(
            #         nn.Linear(input_dim, hidden_dim + i * hidden_dim),
            #         nn.ReLU(),
            #         nn.Linear(hidden_dim + i * hidden_dim, input_dim)
            #     ) for i in range(num_experts)
            # ])
            # self.expert_sizes = torch.arange(hidden_dim, hidden_dim + num_experts * hidden_dim, hidden_dim, dtype=torch.int32)

        elif size_distribution == 'geometric':
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, int(hidden_dim * (1.2 ** i))),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim * (1.2 ** i)), input_dim)
                ) for i in range(num_experts)
            ])
            self.expert_sizes = torch.tensor([int(hidden_dim * (1.2 ** i)) for i in range(num_experts)], dtype=torch.int32)
            
        else:
            raise ValueError("Invalid size distribution. Choose 'uniform', 'arithmetic', or 'geometric'.")
        
        self.gate = SNRAwareGating(input_dim=input_dim, num_experts=num_experts, tau=tau, hard=hard)

    def forward(self, x, snr):
        B, L, D = x.shape
        x_flat = x.view(B * L, D)  # (T, D)
        T = B * L

        gate_scores = self.gate(x, snr)  # (T, N)
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (T, top_k)

        output = torch.zeros_like(x_flat)  # (T, D)
        expert_mask = torch.zeros((T, self.num_experts), device=x.device)  # (T, N)

        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # (T,)
            weight = topk_scores[:, k].unsqueeze(1)  # (T, 1)

            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.sum() == 0:
                    continue
                x_selected = x_flat[mask]  # (N_i, D)
                out = self.experts[i](x_selected)  # (N_i, D)
                output[mask] += out * weight[mask]

                # Mark expert usage in the mask
                expert_mask[mask, i] = 1

        return output.view(B, L, D), gate_scores, expert_mask

class HetereoMoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, num_experts=4, top_k=2, tau=1.0, hard=False, size_distribution='arithmetic'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.hetereo_moe_ffn = HetereoExpertFFN(
            input_dim=d_model,
            hidden_dim=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            tau=tau,
            hard=hard,
            size_distribution=size_distribution,
        )

        self.expert_sizes = self.hetereo_moe_ffn.expert_sizes

    def forward(self, x, snr):
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x2)

        x2, gate_scores, expert_mask  = self.hetereo_moe_ffn(x, snr)
        x = self.norm2(x + x2)

        return x, gate_scores, expert_mask

class HetereoMoETransformer(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2, num_experts=4, top_k=2, dim_feedforward=2048, tau=1.0, hard=False, size_distribution='arithmetic'):
        super().__init__()
        self.layers = nn.ModuleList([
            HetereoMoETransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_experts=num_experts,
                top_k=top_k,
                tau=tau,
                hard=hard,
                size_distribution=size_distribution
            ) for _ in range(num_layers)
        ])

        self.expert_sizes = self.layers[0].expert_sizes

    def forward(self, x, snr):
        all_gate_scores = []
        all_expert_masks = []

        for layer in self.layers:
            x, gate_scores, expert_mask = layer(x, snr)
            all_gate_scores.append(gate_scores)
            all_expert_masks.append(expert_mask)
            
        return x, all_gate_scores, all_expert_masks

    

