import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch, random
from datasets import load_dataset
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import datetime
from transformers import get_cosine_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from semcom_model import HetereoMoE_SemCom

from utils import text_loss, fix_seed, sample_batch, sample_mixed_task_batch, snr_loss, sample_single_task_batch, collate_fn, SST2Dataset, get_test_loader_for_epoch, moe_balancing_loss, moe_balancing_loss_p_penalty, mutual_information_loss, Critic

# Model training configurations
# MODEL_SIZE = 'S'
# NUM_LAYERS = 2
# D_TRANSFORMER = 232
# N_HEADS = 4
# NUM_EXPERTS = 4

MODEL_SIZE = 'M4E'
NUM_LAYERS = 4
D_TRANSFORMER = 412
N_HEADS = 6
NUM_EXPERTS = 4


if __name__ == "__main__":
    # data and model preparation
    rand_seed = 2006
    # fix_seed(rand_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SST-2 (binary sentiment classification)
    # dataset = load_dataset("glue", "sst2", cache_dir='/home/necphy/.cache/huggingface/datasets')
    dataset = load_dataset("glue", "sst2")
    batch_size = 128

    train_dataset = SST2Dataset(dataset['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    model = HetereoMoE_SemCom(num_tasks=2, embed_dim=D_TRANSFORMER, task_dim=8, num_experts=NUM_EXPERTS, size_distribution='arithmetic', transmit_dim=128, num_encd_layer=NUM_LAYERS, num_heads=N_HEADS).to(device)

    mi_critic = Critic(input_dim=2, hidden_dim=12).to(device)
    lambda_mi = 10 #to-do: revert to 10 for possibly better performance

    lr_main = 2e-4, # XL: 0.2e-4
    optimizer_main = torch.optim.AdamW(
        # model.parameters(),
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_main,
        weight_decay=5e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    lr_mi = 1e-4
    optimizer_mi = torch.optim.AdamW(
        mi_critic.parameters(),
        lr=lr_mi,
        weight_decay=5e-4,
    )

    log_val = True

    total_epoch_1 = 250
    total_epoch_2 = 0
    total_epoch = total_epoch_1 + total_epoch_2

    lambda_moe_lb = 2e-4 # previously 5e-4

    # max_steps_per_epoch = 500

    eval_every = 1

    # num_training_steps = len(train_loader) * (total_epoch)
    # num_warmup_steps = int(0.1 * num_training_steps)

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )

    time_start = datetime.datetime.now()
    print(f"Training started at {time_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Training detail: learning rate: 1e-4, weight decay: 5e-3, total epochs: {total_epoch}, batch size: {batch_size}, lambda_moe_lb: {lambda_moe_lb}, seed: {rand_seed}')


    # ------ training phase 1 with perfect channel
    print("Starting training...")
    for epoch in range(total_epoch_1):
        print(f'\n --- Epoch {epoch+1}')
        model.train()
        correct_cls = 0 # correct classification
        total_cls = 0
        recon_loss = []
        cls_loss = []
        moe_lb_loss_arr = []
        total_loss_arr = []
        epoch_expert_mask = []
        mi_loss_arr = []


        for step, (texts, labels) in enumerate(train_loader):
            total_step = step 
            chosen_task = random.choice([0, 1])

            snr = 12.0
            fading = 'none'

            outputs, input_ids, input_lengths, x_complex, y_noisy, gate_scores, expert_masks = model(texts, chosen_task, snr, fading) 
            epoch_expert_mask.append(expert_masks)

            # phase 1: Mutual Information Loss
            mi_loss = mutual_information_loss(x_complex.detach(), y_noisy.detach(), mi_critic) 
            optimizer_mi.zero_grad()
            mi_loss.backward()
            optimizer_mi.step()

            # phase 2: Total Loss
            task_loss = text_loss(
                outputs,
                labels.to(device),
                chosen_task,
                input_ids.to(device),
                input_lengths.to(device)
            )

            moe_lb_loss = moe_balancing_loss_p_penalty(gate_scores, expert_masks, model.expert_sizes.to(device))

            total_loss = task_loss + lambda_moe_lb * moe_lb_loss + lambda_mi*mi_loss.detach()

            optimizer_main.zero_grad()
            total_loss.backward()
            optimizer_main.step()


            if chosen_task == 0:  # Classification
                logits = outputs
                preds = logits.argmax(dim=-1)    

                correct_cls += (preds == labels.to(device)).sum().item()
                total_cls += len(labels)
                cls_loss.append(task_loss.item())

            else:  # Reconstruction
                recon_loss.append(task_loss.item())

            moe_lb_loss_arr.append(moe_lb_loss.item())
            total_loss_arr.append(total_loss.item())
            mi_loss_arr.append(mi_loss.item())


            # # Logging for each model update step
            # if step % 20 == 0:
            #     print(f"----- Step {step} | Task: {chosen_task} | SNR: {snr:.2f} dB | Fading: {fading} | Step Loss: {total_loss:.4f}") 

        # Logging for each epoch
        acc = 100 * correct_cls / total_cls if total_cls > 0 else 0.0
        avg_cls = sum(cls_loss) / len(cls_loss) if len(cls_loss) > 0 else 0.0
        print(f"Task: Classification | Acc: {acc:.2f}% | Avg Loss: {avg_cls:.4f}")
        avg_recon = sum(recon_loss) / len(recon_loss) if len(recon_loss) > 0 else 0.0
        print(f"Task: Reconstruction | Avg Loss: {avg_recon:.4f} ")
        print(f'MoE Balancing Loss: {sum(moe_lb_loss_arr) / len(moe_lb_loss_arr):.4f}')
        print(f"Mutual Information | Avg Loss: {sum(mi_loss_arr) / len(mi_loss_arr):.5f}")

        print(f'Total Loss: {sum(total_loss_arr) / len(total_loss_arr):.4f}')

        # Print expert load distribution for the epoch
        mask_epoch = epoch_expert_mask  # List of expert masks for each batch in the epoch
        num_layers = len(mask_epoch[0])        # Number of layers
        num_experts = mask_epoch[0][0].shape[1]

        # Prepare to aggregate per layer
        usage_per_layer = [torch.zeros(num_experts, device=mask_epoch[0][0].device) for _ in range(num_layers)]

        for batch in mask_epoch:  # batch: list of layer tensors
            for layer_idx, mask in enumerate(batch):
                # mask: (num_tokens_in_batch, num_experts)
                usage_per_layer[layer_idx] += mask.sum(dim=0)

        # Print out expert usage for each layer
        for layer_idx, usage in enumerate(usage_per_layer):
            print(f"Layer {layer_idx}:")
            print(f"-- Expert usage: {[round(x, 2) for x in (usage / usage.sum()).tolist()]}, std: {usage.std().item():.2f}")

        print(f'Time elapsed: {datetime.datetime.now() - time_start}')

        # Epoch-level evaluation
        eval_every = 10  # Evaluate every 1 epochs
        if (epoch + 1) % eval_every == 0 and log_val == True: 
            model.eval()
            bleu_scores = []

            test_loader = get_test_loader_for_epoch(epoch, dataset['validation'], seed=rand_seed, num_samples=3) # return 3 batches, each batch has 1 sample

            for test_step, (texts, labels) in enumerate(test_loader):
                snr = 12.0
                fading = 'none'
                chosen_task = 1

                with torch.no_grad():
                    outputs, input_ids, input_lengths, semantic_encoded, semantic_decoded, gate_scores, expert_masks = model(texts, chosen_task, snr, fading) 

                    pred_ids_batch = outputs.argmax(dim=-1).cpu().tolist()

                for i in range(len(texts)):
                    # target_len = input_lengths[i].item()

                    # pred_ids = outputs[i].cpu().squeeze().tolist()
                    # tgt_ids = input_ids[i][1:1 + len(pred_ids)].cpu().tolist()

                    target_len = input_lengths[i].item()

                    # Directly align without shift
                    pred_ids = pred_ids_batch[i]
                    tgt_ids = input_ids[i][:len(pred_ids)].cpu().tolist()


                    pred_text = model.text_encoder.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    target_text = model.text_encoder.tokenizer.decode(tgt_ids, skip_special_tokens=True)

                    bleu = sentence_bleu([word_tokenize(target_text)], word_tokenize(pred_text), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
                    bleu_scores.append(bleu)

                    print(f'Example {test_step + 1} ---')
                    print("Original text:", target_text)
                    print("Reconstructed text:", pred_text)
                    print(f'Original IDs: {tgt_ids}')
                    print(f'Predicted IDs: {pred_ids}')
                    print("BLEU Score:", f"{bleu:.4f}")

            # avg_bleu = sum(bleu_scores) / len(bleu_scores)
            # print(f"[Eval @ Epoch {epoch+1}] Avg BLEU Score: {avg_bleu:.4f}")
            model.train()


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"./checkpoints/HMoE_size{MODEL_SIZE}_{NUM_LAYERS}_{NUM_EXPERTS}_{N_HEADS}_{D_TRANSFORMER}_{timestamp}.pt")

    # --------------------
    # Testing (BLEU Score for Reconstruction)
    # --------------------
    print("\n--- Final Test BLEU Score ---")
    model.eval()
    all_bleu_scores = []
    printed_examples = 0
    printed_this_batch = False 

    test_dataset = SST2Dataset(dataset['validation'])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    with torch.no_grad():
        for texts, labels in test_loader:
            bleu_scores = []

            fading = random.choice(['none', 'rayleigh', 'rician'])
            snr = random.uniform(0.0, 20.0)

            outputs, input_ids, input_lengths, _, _, _, _ = model(texts, 1, snr, fading) 

            batch_output_preds = outputs.argmax(dim=-1).cpu().tolist()

            pred_ids_batch = batch_output_preds
            tgt_ids_batch = input_ids[:, :len(pred_ids_batch[0])].cpu().tolist()

            # pred_text = model.text_encoder.tokenizer.decode(pred_ids, skip_special_tokens=True)
            # target_text = model.text_encoder.tokenizer.decode(tgt_ids, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(pred_ids_batch, skip_special_tokens=True)
            target_texts = tokenizer.batch_decode(tgt_ids_batch, skip_special_tokens=True)

            for pred_text, target_text in zip(pred_texts, target_texts):
                bleu = sentence_bleu([word_tokenize(target_text)], word_tokenize(pred_text), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
                bleu_scores.append(bleu)

            all_bleu_scores.append(sum(bleu_scores) / len(bleu_scores))

    print(f"Avg BLEU Score: {sum(all_bleu_scores)/len(all_bleu_scores):.4f}")


# nohup python -u main_hetereoMoE.py > ./log/HetereoMoE_sizeM4E_$(date +%Y%m%d_%H%M%S).log 2>&1 & 



    






    