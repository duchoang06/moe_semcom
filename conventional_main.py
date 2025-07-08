import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch, random, math
from datasets import load_dataset, load_from_disk
import torch.nn.functional as F
import torch.nn as nn
from joblib import Parallel, delayed


from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import datetime
from transformers import get_cosine_schedule_with_warmup
from transformers import BertTokenizer

from torch.utils.data import Dataset, DataLoader

from conventional_model import Conventional_Com

from utils import text_loss, fix_seed, sample_mixed_task_batch, sample_single_task_batch, collate_fn, SST2Dataset, get_test_loader_for_epoch, moe_balancing_loss_p_penalty, mutual_information_loss, Critic, QQPPromptDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix_seed(42)

    dataset = load_dataset('glue' , 'qqp')
    batch_size = 256

    # train_dataset = QQPPromptDataset(dataset['train'])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    full_train_dataset = dataset['train']
    small_indices = random.sample(range(len(full_train_dataset)), len(full_train_dataset) // 5)
    small_train_dataset = full_train_dataset.select(small_indices)
    train_dataset = QQPPromptDataset(small_train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    test_dataset = QQPPromptDataset(dataset['validation'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Conventional_Com(emded_dim=512).to(device) # 2 means 2 classes

    lr_main = 5e-4
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_main,
        # weight_decay=1e-2
    )

    total_epoch = 1
    criterion = nn.CrossEntropyLoss()


    time_start = datetime.datetime.now()
    print(f"Training started at {time_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Training detail: learning rate: {lr_main}, weight decay: 1e-2, total epochs: {total_epoch}, batch size: {batch_size}')

    for epoch in range(total_epoch):
        print(f'\n --- Epoch {epoch+1}')

        model.train() 
        correct_cls = 0
        total_cls = 0
        cls_loss = []

        for texts, labels in train_loader:
            snr = 12.0
            fading = 'none'

            logits = model(texts, snr=snr, fading=fading)

            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cls_loss.append(loss.item())

            # prediction
            preds = logits.argmax(dim=-1)
            correct_cls += (preds == labels.to(device)).sum().item()
            total_cls += len(labels)

            print(f'batch acc : {(preds == labels.to(device)).sum().item()/batch_size * 100:.2f}%')
            print(f'time now: {datetime.datetime.now()}')


        acc = 100 * correct_cls / total_cls if total_cls > 0 else 0
        avg_loss = sum(cls_loss) / len(cls_loss) if len(cls_loss) > 0 else 0.0
        print(f"Task: Classification | Acc: {acc:.2f}% | Avg Loss: {avg_loss:.3f}")

        print(f'Time elapsed: {datetime.datetime.now() - time_start}')

        eval_epoch = 1 
        if (epoch + 1) % eval_epoch == 0:
            model.eval()
            correct_cls = 0
            total_cls = 0

            with torch.no_grad():
                for texts, labels in test_loader:
                    snr = 12.0
                    fading = 'none'

                    logits = model(texts, snr=snr, fading=fading)

                    # prediction
                    preds = logits.argmax(dim=-1)
                    correct_cls += (preds == labels.to(device)).sum().item()
                    total_cls += len(labels)

            acc = 100 * correct_cls / total_cls if total_cls > 0 else 0
            print(f"Eval Classification | Acc: {acc:.2f}%")

        model.train()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f"./checkpoints_new/Conv_Bert_{timestamp}.pt")

# nohup python -u conventional_main.py > ./log/conv_bert.log 2>&1 &

