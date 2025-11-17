# fig1: acc vs snr, lines: models

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch, random
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from semcom_model import MoE_SemCom, Transformer_SemCom, HetereoMoE_SemCom 
from conventional_model import Conventional_Com

from utils import collate_fn, SST2Dataset, fix_seed, QQPPromptDataset
import datetime

plt_paras = {'ytick.color' : 'black',
          'xtick.color' : 'black',
          'axes.labelcolor' : 'black',
          'axes.edgecolor' : 'black',
          'text.usetex' : True,
          'font.family' : 'serif',
          'font.serif' : 'Computer Modern',
          'font.size': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
        #   'axes.labelsize': 16,
        #   'figure.labelsize': 16,
          'figure.autolayout': True,
          'legend.fontsize': 13,
          'legend.loc': 'best',
          'axes.grid': True,   
          'axes.grid.axis': 'both',    
          'axes.grid.which': 'both',
          'grid.alpha': 0.7,  # Adjust the opacity (0.0 to 1.0)

          'grid.linestyle': 'dotted',
        #   'lines.linewidth': 2,
        #   'lines.markersize': 7,  # Adjust the marker size as needed

        }

plt.rcParams.update(plt_paras)


# Setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# fix_seed(1997) # 2006
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
# model_dense = Transformer_SemCom(num_tasks=2, embed_dim=632, task_dim=8, num_encd_layer=6, transmit_dim=128, num_heads=8).to(device)

# model_hetereo = HetereoMoE_SemCom(num_tasks=2, embed_dim=632, task_dim=8, num_experts=12, size_distribution='arithmetic', transmit_dim=128, num_encd_layer=6, num_heads=8).to(device) 

# model_moe = MoE_SemCom(num_tasks=2, embed_dim=632, task_dim=8, num_experts=12, transmit_dim=128, num_encd_layer=6, num_heads=8).to(device) 

model_conv = Conventional_Com(emded_dim=512).to(device) 


# model_moe.load_state_dict(torch.load("checkpoints/Dense_snr12_20250604_055257.pt"))

# model_dense.load_state_dict(torch.load("checkpoints_new/Dense_sizeL_6_1_8_632_20250701_011126.pt", weights_only=True))
# model_hetereo.load_state_dict(torch.load("checkpoints_new/HMoE_sizeL_6_12_8_632_20250702_084802.pt", weights_only=True)) 
# model_moe.load_state_dict(torch.load("checkpoints_new/MoE_sizeL_20250709_015013.pt", weights_only=True)) 
model_conv.load_state_dict(torch.load("checkpoints_new/Conv_Bert_20250708_224213.pt", weights_only=True))

# model_moe.eval()
# model_dense.eval()
# model_hetereo.eval()
# model_moe.eval()
model_conv.eval()

# Load data
dataset = load_dataset("glue", "qqp")
batch_size = 512


# test_dataset = SST2Dataset(dataset['validation'])
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

full_test_dataset = dataset["validation"]
subset_indices = random.sample(range(len(full_test_dataset)), len(full_test_dataset) // 40)
small_test_dataset = full_test_dataset.select(subset_indices)
test_dataset = QQPPromptDataset(small_test_dataset)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size, 
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
)


snr_range = np.arange(-8, 22, 2)
fading_range = ['none', 'rayleigh', 'rician']

# Results dicts
results = {
    # 'dense': {'accuracy': {f: [] for f in fading_range}, 'bleu': {f: [] for f in fading_range}},
    # 'moe': {'accuracy': {f: [] for f in fading_range}, 'bleu': {f: [] for f in fading_range}},
    # 'hmoe': {'accuracy': {f: [] for f in fading_range}, 'bleu': {f: [] for f in fading_range}},
    'conv': {'accuracy': {f: [] for f in fading_range}, 'bleu': {f: [] for f in fading_range}},
}


time_start = datetime.datetime.now()
print(f'Starting evaluation at {time_start.strftime("%Y-%m-%d %H:%M:%S")}')

# Main evaluation loop
for snr in snr_range:
    for fading in fading_range:
        metrics = {arch: {'correct': 0, 'total': 0, 'bleu_scores': []} for arch in results.keys()}

        for batch_idx, (texts, labels) in enumerate(test_loader):
            for arch_name, model in [('conv', model_conv),]:
                                    #  ('conv', model_conv)]: 

                for task in [0]:
                    with torch.no_grad():
                        if arch_name == 'dense':
                            outputs, input_ids, _, _, _ = model(texts, task, float(snr), fading, rician_k=4.0, modal=True)
                        elif arch_name == 'conv':
                            outputs = model(texts, float(snr), fading, rician_k=4.0, modal=False)
                        elif arch_name == 'moe':
                            outputs, input_ids, _, _, _, _, _ = model(texts, task, float(snr), fading, rician_k=4.0, modal=True)
                        else:
                            outputs, input_ids, _, _, _, _, _ = model(texts, task, float(snr), fading, rician_k=4.0, modal=False)

                        if task == 0:  # Classification
                            preds = outputs.argmax(dim=-1)
                            metrics[arch_name]['correct'] += (preds == labels.to(device)).sum().item()
                            metrics[arch_name]['total'] += labels.size(0)

                        elif task == 1:  # Reconstruction
                            pred_ids_batch = outputs.argmax(dim=-1).cpu().tolist()

                            tgt_ids_batch = input_ids[:, :len(pred_ids_batch[0])].cpu().tolist() 

                            pred_texts = tokenizer.batch_decode(pred_ids_batch, skip_special_tokens=False)
                            target_texts = tokenizer.batch_decode(tgt_ids_batch, skip_special_tokens=False)

                            for pred_text, target_text in zip(pred_texts, target_texts):
                                bleu = sentence_bleu(
                                    [word_tokenize(target_text)],
                                    word_tokenize(pred_text),
                                    weights=(1, 0, 0, 0),
                                    smoothing_function=SmoothingFunction().method4
                                )
                                metrics[arch_name]['bleu_scores'].append(bleu)

            print(f'Processed batch {batch_idx + 1}/{len(test_loader)}, time: {datetime.datetime.now()}')

        # Aggregate
        for arch_name in results.keys():
            acc = metrics[arch_name]['correct'] / metrics[arch_name]['total'] if metrics[arch_name]['total'] > 0 else 0
            bleu = np.mean(metrics[arch_name]['bleu_scores'])

            print(f'accuracy: {acc*100:2f}%')

            results[arch_name]['accuracy'][fading].append(acc*100)  
            results[arch_name]['bleu'][fading].append(bleu)

        print(f'Done for SNR={snr} dB, Fading={fading}, time elapsed: {datetime.datetime.now() - time_start}')

# Save results with pickle
import pickle
with open('./figures/results_conv_bert.pkl', 'wb') as f:
    pickle.dump(results, f)


# Plotting: 3 figures, one per fading
# model_types = ['dense', 'moe', 'hetereoMoE']  # NEW
# model_types = ['hmoe', 'dense', 'moe', ]  
# model_configs = {
#     'dense': {'label': 'Dense', 'color': 'blue', 'marker': 'o', 'linestyle': '-'},
#     'moe': {'label': 'MoE', 'color': 'orange', 'marker': 's', 'linestyle': '--'},
#     'hmoe': {'label': 'HMoE', 'color': 'green', 'marker': '^', 'linestyle': '-.'},
#     'conv': {'label': 'UTF-8 & LDPC', 'color': 'red', 'marker': 'd', 'linestyle': ':'},
# }

# for fading in fading_range:
#     fig, axs = plt.subplots()

#     # Accuracy plot
#     for model in model_types:
#         label = model_configs[model]['label']
#         color = model_configs[model]['color']
#         marker = model_configs[model]['marker']
#         linestyle = model_configs[model]['linestyle']

#         axs.plot(snr_range, results[model]['accuracy'][fading], label=label, marker=marker, color=color, linestyle=linestyle)

#     # axs.set_title(f'Accuracy vs SNR ({fading} fading)')
#     axs.set_xlabel(r'SNR (dB)')
#     axs.set_ylabel(r'Accuracy (\%)')
#     axs.legend(loc='best', fontsize=12)
#     axs.grid(True)

#     axs.set_xticks([-8, -4, 0, 4, 8, 12, 16, 20])  # Set specific x-ticks


    # # BLEU plot
    # for model in model_types:
    #     axs[1].plot(snr_range, results[model]['bleu'][fading], label=model)
    # axs[1].set_title(f'BLEU Score vs SNR ({fading} fading)')
    # axs[1].set_xlabel('SNR (dB)')
    # axs[1].set_ylabel('BLEU Score')
    # axs[1].legend()
    # axs[1].grid(True)

    # plt.savefig(f'./figures/acc_{fading}_lite.pdf', bbox_inches='tight')

    # plt.tight_layout()
    # plt.show()

# nohup python -u conv_sim.py > ./log/conv_sim.log 2>&1 &