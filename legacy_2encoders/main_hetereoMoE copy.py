import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'

import numpy as np
import torch, random
from datasets import load_dataset
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import datetime
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader

from semcom_model import HetereoMoE_SemCom
from utils import text_loss, fix_seed, sample_batch, sample_mixed_task_batch, snr_loss, sample_single_task_batch, collate_fn, SST2Dataset, get_test_loader_for_epoch, moe_balancing_loss


#to-do: dense model without wireless -> dense model with wireless -> expert training with wireless 
if __name__ == "__main__":
    # data and model preparation
    rand_seed = 2006
    fix_seed(rand_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SST-2 (binary sentiment classification)
    dataset = load_dataset("glue", "sst2", cache_dir='/home/necphy/.cache/huggingface/datasets')
    batch_size = 128

    train_dataset = SST2Dataset(dataset['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # test_dataset = SST2Dataset(dataset['validation'])
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = HetereoMoE_SemCom(num_tasks=2, embed_dim=256, task_dim=8, num_experts=8, size_distribution='arithmetic', transmit_dim=128, num_encd_layer=4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
    lr_gamma = 0.95
    log_val = False

    total_epoch = 200
    # lambda_moe_lb = 2e-4 # previously 5e-4
    lambda_moe_lb = 1

    # max_steps_per_epoch = 500

    eval_every = 10 


    num_training_steps = len(train_loader) * (total_epoch)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ------ training phase 1 with perfect channel
    print("Starting training...")
    for epoch in range(total_epoch // 2):
        print(f'\n --- Epoch {epoch+1}')
        model.train()
        correct_cls = 0 # correct classification
        total_cls = 0
        recon_loss = []
        cls_loss = []
        moe_lb_loss_arr = []
        total_loss_arr = []
        epoch_expert_mask = []

        for step, (texts, labels) in enumerate(train_loader):
            total_step = step 
            chosen_task = random.choice([0, 1])

            # task_ids = torch.tensor([chosen_task for _ in range(len(texts))])

            # snr = random.uniform(15, 25) # dB
            # fading = random.choice(['none'])

            snr = 30
            fading = 'none'

            outputs, input_ids, input_lengths, semantic_encoded, semantic_decoded, gate_scores, expert_masks = model(texts, chosen_task, snr, fading) 
            epoch_expert_mask.append(expert_masks)

            #to-do: log expert load distribution in each epoch here
            # for encoder_layer, expert_mask in enumerate(expert_masks):
            #     print(f"Layer {encoder_layer} - Expert Load Distribution: {expert_mask.sum(dim=0).cpu().numpy()}")

            task_loss = text_loss(
                outputs,
                labels.to(device),
                chosen_task,
                input_ids.to(device),
                input_lengths.to(device)
            )

            moe_lb_loss = moe_balancing_loss(gate_scores, expert_masks, model.expert_sizes.to(device))

            total_loss = task_loss + lambda_moe_lb * moe_lb_loss

            total_loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

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
            print(f"-- Expert usage: {usage.tolist()}, std: {usage.std().item():.2f}")


        # Epoch-level evaluation
        if (epoch + 1) % eval_every == 0 and log_val == True: 
            model.eval()
            bleu_scores = []

            test_loader = get_test_loader_for_epoch(epoch, dataset['validation'], seed=rand_seed, num_samples=3) # return 3 batches, each batch has 1 sample

            for test_step, (texts, labels) in enumerate(test_loader):
                snr = 30
                fading = 'none'
                chosen_task = 1

                with torch.no_grad():
                    outputs, input_ids, input_lengths, semantic_encoded, semantic_decoded, gate_scores, expert_masks = model(texts, chosen_task, snr, fading) 

                for i in range(len(texts)):
                    target_len = input_lengths[i].item()

                    pred_ids = outputs[i].cpu().squeeze().tolist()
                    tgt_ids = input_ids[i][1:1 + len(pred_ids)].cpu().tolist()

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


    # ---- training phase 2: fine tuning with varied channel conditions   
    for epoch in range(total_epoch // 2, total_epoch):
        print(f'\n --- Epoch {epoch+1}')
        model.train()
        correct_cls = 0 # correct classification
        total_cls = 0
        recon_loss = []
        cls_loss = []
        moe_lb_loss_arr = []
        total_loss_arr = []

        for step, (texts, labels) in enumerate(train_loader):
            total_step = step 

            chosen_task = random.choice([0, 1])

            # task_ids = torch.tensor([chosen_task for _ in range(len(texts))])

            snr = random.uniform(-5, 25) # dB
            fading = random.choice(['none', 'rayleigh', 'rician'])

            outputs, input_ids, input_lengths, semantic_encoded, semantic_decoded, gate_scores, expert_masks = model(texts, chosen_task, snr, fading) 
            epoch_expert_mask.append(expert_masks)


            task_loss = text_loss(
                outputs,
                labels.to(device),
                chosen_task,
                input_ids.to(device),
                input_lengths.to(device)
            )

            moe_lb_loss = moe_balancing_loss(gate_scores, expert_masks, model.expert_sizes.to(device))

            total_loss = task_loss + lambda_moe_lb * moe_lb_loss

            total_loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

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
            print(f"-- Expert usage: {usage.tolist()}, std: {usage.std().item():.2f}")

        # Epoch-level evaluation
        if (epoch + 1) % eval_every == 0 and log_val == True:
            model.eval()
            bleu_scores = []

            test_loader = get_test_loader_for_epoch(epoch, dataset['validation'], seed=rand_seed, num_samples=3)  # return 3 batches, each batch has 1 sample

            for test_step, (texts, labels) in enumerate(test_loader):
                snr = 30
                fading = 'none'
                chosen_task = 1

                with torch.no_grad():
                    outputs, input_ids, input_lengths, semantic_encoded, semantic_decoded, gate_scores, expert_masks = model(texts, chosen_task, snr, fading) 

                for i in range(len(texts)):
                    target_len = input_lengths[i].item()

                    pred_ids = outputs[i].cpu().squeeze().tolist()
                    tgt_ids = input_ids[i][1:1 + len(pred_ids)].cpu().tolist()

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
    torch.save(model.state_dict(), f"./checkpoints/HetereoMoE_encdlayer_{timestamp}.pt")

    # --------------------
    # Testing (BLEU Score for Reconstruction)
    # --------------------
    print("\n--- Final Test BLEU Score ---")
    model.eval()
    bleu_scores = []
    printed_examples = 0
    printed_this_batch = False 

    test_dataset = SST2Dataset(dataset['validation'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    with torch.no_grad():
        for texts, labels in test_loader:

            fading = random.choice(['none', 'rayleigh', 'rician'])
            snr = random.uniform(-5, 25)

            outputs, input_ids, input_lengths, semantic_encoded, semantic_decoded, gate_scores, expert_masks = model(texts, 1, snr, fading) 

            for i in range(len(texts)):
                target_len = input_lengths[i].item()
                pred_ids = outputs[i].cpu().squeeze().tolist()
                tgt_ids = input_ids[i][1:1 + len(pred_ids)].cpu().tolist()

                pred_text = model.text_encoder.tokenizer.decode(pred_ids, skip_special_tokens=True)
                target_text = model.text_encoder.tokenizer.decode(tgt_ids, skip_special_tokens=True)

                bleu = sentence_bleu([word_tokenize(target_text)], word_tokenize(pred_text), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
                bleu_scores.append(bleu)

                if printed_examples < 5 and not printed_this_batch:
                    print(f'--- Example {printed_examples + 1} ---, SNR: {snr:.2f} dB, Fading: {fading}')
                    print("Original:", target_text)
                    print("Reconstructed:", pred_text)
                    print("BLEU Score:", f"{bleu:.4f}")

                    printed_examples += 1
                    printed_this_batch = True
            
            printed_this_batch = False  # Reset for the next batch

    print(f"Avg BLEU Score: {sum(bleu_scores)/len(bleu_scores):.4f}")


# nohup python -u main_hetereoMoE.py > ./log/HetereoMoE_encdlayer_$(date +%Y%m%d_%H%M%S).log 2>&1 & 



    






    