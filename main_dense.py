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
from transformers import BertTokenizer

from torch.utils.data import Dataset, DataLoader

from semcom_model import Transformer_SemCom

from utils import text_loss, fix_seed, sample_mixed_task_batch, sample_single_task_batch, collate_fn, SST2Dataset, get_test_loader_for_epoch, moe_balancing_loss_p_penalty, mutual_information_loss, Critic


# MAIN TRAINING PARAS:
# SMALL Dense:
# MODEL_SIZE = 'S'  # 'S', 'M', 'L'
# NUM_LAYERS = 2
# D_TRANSFORMER = 232
# N_HEADS = 4
# NUM_EXPERTS = 1

# M Dense:
# MODEL_SIZE = 'M'  # 'S', 'M', 'L'
# NUM_LAYERS = 4
# D_TRANSFORMER = 412
# N_HEADS = 6
# NUM_EXPERTS = 1

# L Dense:
MODEL_SIZE = 'L'  # 'S', 'M', 'L'
NUM_LAYERS = 6
D_TRANSFORMER = 632
N_HEADS = 8
NUM_EXPERTS = 1


if __name__ == "__main__":
    # data and model preparation
    rand_seed = 2231
    # fix_seed(rand_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SST-2 (binary sentiment classification)
    # dataset = load_dataset("glue", "sst2", cache_dir='/home/necphy/.cache/huggingface/datasets')
    dataset = load_dataset("glue", "sst2")

    batch_size = 128

    train_dataset = SST2Dataset(dataset['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # test_dataset = SST2Dataset(dataset['validation'])
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = Transformer_SemCom(num_tasks=2, embed_dim=D_TRANSFORMER, task_dim=8, num_encd_layer=NUM_LAYERS, transmit_dim=128, num_heads=N_HEADS).to(device)

    mi_critic = Critic(input_dim=2, hidden_dim=12).to(device)
    lambda_mi = 10 #to-do: revert to 10 for possibly better performance

    lr_main = 0.3e-4
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
    
    # lr_gamma = 0.95
    log_val = True

    # max_steps_per_epoch = 500
    total_epoch_1 = 350
    total_epoch_2 = 0
    total_epoch = total_epoch_1 + total_epoch_2

    # num_training_steps_1 = len(train_loader) * (total_epoch_1)
    # num_warmup_steps_1 = int(0.1 * num_training_steps_1)
    # scheduler_mi = get_cosine_schedule_with_warmup(
    #     optimizer_mi,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps_1,
    # )

    # num_training_steps_2 = len(train_loader) * (total_epoch_1)
    # num_warmup_steps_2 = int(0.2 * num_training_steps_2)
    # scheduler_main = get_cosine_schedule_with_warmup(
    #     optimizer_main,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps_2,
    # )

    time_start = datetime.datetime.now()
    print(f"Training started at {time_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Training detail: learning rate: 2e-4, weight decay: 5e-4, total epochs: {total_epoch}, batch size: {batch_size}, seed: {rand_seed}')


    # ------ training phase 1 with perfect channel
    print("Starting training...")
    for epoch in range(total_epoch_1):
        print(f'\n --- Epoch {epoch+1}')
        model.train()
        mi_critic.train()

        correct_cls = 0 # correct classification
        total_cls = 0
        recon_loss = []
        cls_loss = []
        mi_loss_arr = []
        total_loss_arr = []

        for step, (texts, labels) in enumerate(train_loader):
            chosen_task = random.choice([0, 1])
            # chosen_task = 1
            fading = 'none'  # 'none', 'rayleigh', 'rician'
            snr = 12.0

            # snr = random.uniform(15, 25) # dB
            # fading = random.choice(['none', 'rayleigh', 'rician'])

            outputs, input_ids, input_lengths, x_complex, y_noisy = model(texts, chosen_task, snr, fading) 

            # phase 1: Mutual Information Loss
            mi_loss = mutual_information_loss(x_complex.detach(), y_noisy.detach(), mi_critic) 
            optimizer_mi.zero_grad()
            mi_loss.backward()
            optimizer_mi.step()
            # scheduler_mi.step()

            # phase 2: Total Loss
            task_loss = text_loss(
                outputs,
                labels.to(device),
                chosen_task,
                input_ids.to(device),
                input_lengths.to(device)
            )

            total_loss = task_loss + lambda_mi*mi_loss.detach()
            optimizer_main.zero_grad()
            total_loss.backward()
            optimizer_main.step()
            # scheduler_main.step()

            if chosen_task == 0:  # Classification
                logits = outputs
                preds = logits.argmax(dim=-1)    

                correct_cls += (preds == labels.to(device)).sum().item()
                total_cls += len(labels)
                cls_loss.append(task_loss.item())

            else:  # Reconstruction
                recon_loss.append(task_loss.item())

            mi_loss_arr.append(mi_loss.item())
            total_loss_arr.append(total_loss.item())


        # Logging for each epoch
        acc = 100 * correct_cls / total_cls if total_cls > 0 else 0.0
        avg_cls = sum(cls_loss) / len(cls_loss) if len(cls_loss) > 0 else 0.0
        print(f"Task: Classification | Acc: {acc:.2f}% | Avg Loss: {avg_cls:.3f}")
        avg_recon = sum(recon_loss) / len(recon_loss) if len(recon_loss) > 0 else 0.0
        print(f"Task: Reconstruction | Avg Loss: {avg_recon:.3f} ")
        print(f"Mutual Information | Avg Loss: {sum(mi_loss_arr) / len(mi_loss_arr):.5f}")
        print(f"Total Loss | Avg Loss: {sum(total_loss_arr) / len(total_loss_arr):.3f}")

        print(f'Time elapsed: {datetime.datetime.now() - time_start}')


        # Epoch-level evaluation
        eval_every = 10  # Evaluate every 1 epochs
        if (epoch + 1) % eval_every == 0 and log_val:
            model.eval()
            bleu_scores = []

            test_loader = get_test_loader_for_epoch(epoch, dataset['validation'], seed=rand_seed, num_samples=3) # return 3 batches, each batch has 1 sample

            for test_step, (texts, labels) in enumerate(test_loader):
                snr = 12.0
                fading = 'none'
                chosen_task = 1

                with torch.no_grad():
                    outputs, input_ids, input_lengths, _, _ = model(texts, chosen_task, snr=snr, fading=fading)

                    pred_ids_batch = outputs.argmax(dim=-1).cpu().tolist()

                for i in range(len(texts)):
                    target_len = input_lengths[i].item()

                    pred_ids = pred_ids_batch[i]
                    tgt_ids = input_ids[i][:len(pred_ids)].cpu().tolist()

                    pred_text = model.text_encoder.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    target_text = model.text_encoder.tokenizer.decode(tgt_ids, skip_special_tokens=True)

                    bleu = sentence_bleu([word_tokenize(target_text)], word_tokenize(pred_text), smoothing_function=SmoothingFunction().method4, weights=(1, 0, 0, 0),)
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


    # # ---- training phase 2 
    # for param in mi_critic.parameters(): # freeze the MI critic model
    #     param.requires_grad = False 

    # snr_stages = [12.0, 8.0, 6.0]
    # snr_stage_len = total_epoch_2 // len(snr_stages)

    # for epoch in range(total_epoch_2):
    #     print(f'\n --- Epoch {epoch+1}')
    #     model.train()

    #     correct_cls = 0 # correct classification
    #     total_cls = 0
    #     recon_loss = []
    #     cls_loss = []
    #     mi_loss_arr = []
    #     total_loss_arr = []
    #     const_loss_arr = []

    #     for step, (texts, labels) in enumerate(train_loader):
    #         chosen_task = random.choice([0, 1])

    #         # snr = random.uniform(-5, 20) # dB
    #         # fading = random.choice(['none', 'rayleigh', 'rician'])
    #         stage = min(epoch // snr_stage_len, len(snr_stages) - 1)  # ensure doesn't go out of bounds
    #         snr = snr_stages[stage]

    #         fading = 'none'

    #         outputs, input_ids, input_lengths, x_transmitted, y_noisy = model(texts, chosen_task, snr, fading) 

    #         # Consistency loss
    #         # snr_2 = random.choice([snr, snr-1, snr-2])
    #         # outputs_2, _, _, _, _ = model(texts, chosen_task, snr_2 , fading)
    #         # consistency_loss = F.mse_loss(outputs, outputs_2.detach())


    #         task_loss = text_loss(
    #             outputs,
    #             labels.to(device),
    #             chosen_task,
    #             input_ids.to(device),
    #             input_lengths.to(device)
    #         )

    #         mi_loss = mutual_information_loss(x_transmitted, y_noisy, mi_critic)


    #         total_loss = task_loss + lambda_mi * mi_loss 

    #         total_loss.backward()

    #         optimizer_main.step()
    #         # scheduler_main.step()
    #         optimizer_main.zero_grad()

    #         if chosen_task == 0:  # Classification
    #             logits = outputs
    #             preds = logits.argmax(dim=-1)    

    #             correct_cls += (preds == labels.to(device)).sum().item()
    #             total_cls += len(labels)
    #             cls_loss.append(task_loss.item())

    #         else:  # Reconstruction
    #             recon_loss.append(task_loss.item())

    #         mi_loss_arr.append(mi_loss.item())
    #         total_loss_arr.append(total_loss.item())

    #     # Logging for each epoch
    #     acc = 100 * correct_cls / total_cls if total_cls > 0 else 0.0
    #     avg_cls = sum(cls_loss) / len(cls_loss) if len(cls_loss) > 0 else 0.0
    #     print(f"Task: Classification | Acc: {acc:.2f}% | Avg Loss: {avg_cls:.3f}")
    #     print(f"Task: Reconstruction | Avg Loss: {sum(recon_loss) / len(recon_loss):.3f} ")
    #     print(f"Mutual Information | Avg Loss: {sum(mi_loss_arr) / len(mi_loss_arr):.5f}")
    #     print(f"Total Loss | Avg Loss: {sum(total_loss_arr) / len(total_loss_arr):.3f}")

    #     print(f'Time elapsed: {datetime.datetime.now() - time_start}')


    #     # Epoch-level evaluation
    #     eval_every = 10  # Evaluate every 1 epochs
    #     if (epoch + 1) % eval_every == 0 and log_val:
    #         model.eval()
    #         bleu_scores = []

    #         test_loader = get_test_loader_for_epoch(epoch, dataset['validation'], seed=rand_seed, num_samples=3)  # return 3 batches, each batch has 1 sample

    #         for test_step, (texts, labels) in enumerate(test_loader):
    #             snr = 12.0
    #             fading = 'none'
    #             chosen_task = 1

    #             with torch.no_grad():
    #                 outputs, input_ids, input_lengths, _, _ = model(texts, chosen_task, snr=snr, fading=fading)

    #                 pred_ids_batch = outputs.argmax(dim=-1).cpu().tolist()

    #             for i in range(len(texts)):
    #                 target_len = input_lengths[i].item()

    #                 pred_ids = pred_ids_batch[i]
    #                 tgt_ids = input_ids[i][: len(pred_ids)].cpu().tolist()

    #                 pred_text = model.text_encoder.tokenizer.decode(pred_ids, skip_special_tokens=True)
    #                 target_text = model.text_encoder.tokenizer.decode(tgt_ids, skip_special_tokens=True)

    #                 bleu = sentence_bleu([word_tokenize(target_text)], word_tokenize(pred_text), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
    #                 bleu_scores.append(bleu)

    #                 print(f'Example {test_step + 1} ---')
    #                 print("Original text:", target_text)
    #                 print("Reconstructed text:", pred_text)
    #                 print(f'Original IDs: {tgt_ids}')
    #                 print(f'Predicted IDs: {pred_ids}')
    #                 print("BLEU Score:", f"{bleu:.4f}")

    #         # avg_bleu = sum(bleu_scores) / len(bleu_scores)
    #         # print(f"[Eval @ Epoch {epoch+1}] Avg BLEU Score: {avg_bleu:.4f}")
    #         model.train()


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"./checkpoints/Dense_size{MODEL_SIZE}_{NUM_LAYERS}_{NUM_EXPERTS}_{N_HEADS}_{D_TRANSFORMER}_{timestamp}.pt")

    # --------------------
    # Testing (BLEU Score for Reconstruction)
    # --------------------
    print("\n--- Final Test BLEU Score ---")
    model.eval()
    bleu_scores = []
    printed_examples = 0
    printed_this_batch = False 
    all_bleu_scores = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    test_dataset = SST2Dataset(dataset['validation'])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    with torch.no_grad():
        for texts, labels in test_loader:

            fading = random.choice(['none', 'rayleigh', 'rician'])
            snr = random.uniform(0.0, 20.0)

            # task_ids = torch.full((len(texts),), 1, dtype=torch.long)
            outputs, input_ids, input_lengths, _, _ = model(texts, 1, snr=snr, fading=fading)

            batch_output_preds = outputs.argmax(dim=-1).cpu().tolist()

            pred_ids_batch = batch_output_preds
            tgt_ids_batch = input_ids[:, :len(pred_ids_batch[0])].cpu().tolist()


            pred_texts = tokenizer.batch_decode(pred_ids_batch, skip_special_tokens=True)
            target_texts = tokenizer.batch_decode(tgt_ids_batch, skip_special_tokens=True)

            for pred_text, target_text in zip(pred_texts, target_texts):
                bleu = sentence_bleu([word_tokenize(target_text)], word_tokenize(pred_text), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
                bleu_scores.append(bleu)

            all_bleu_scores.append(sum(bleu_scores) / len(bleu_scores))

    print(f"Avg BLEU Score: {sum(all_bleu_scores) / len(all_bleu_scores):.4f}")
    print(f'Training lasted {datetime.datetime.now() - time_start}')


# nohup python -u main_dense.py > ./log/Dense_sizeL_$(date +%Y%m%d_%H%M%S).log 2>&1 &
