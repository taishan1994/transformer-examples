from ast import In
import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from model import TransformerSeq2Seq


class DLDataset(Dataset):
    def __init__(self, args):
        self.data = []
        with open(args.src_path, "r", encoding="utf-8") as fp:
            srcs = fp.read().strip().split("\n")
        with open(args.trg_path, "r", encoding="utf-8") as fp:
            trgs = fp.read().strip().split("\n")
        for src, trg in zip(srcs, trgs):
            src = "".join(src.strip().split(" "))
            trg = "".join(trg.strip().split(" "))
            self.data.append((src, trg))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class Collate:
    def __init__(self, args):
        self.src_seq_len = args.src_seq_len
        self.trg_seq_len = args.trg_seq_len
        self.token2id = args.token2id
        self.id2token = args.id2token
        self.eos_id = args.eos_id
        self.sos_id = args.sos_id

    def collate_fn(self, batch):
        src_ids = []
        trg_ids = []
        label_ids = []
        for i, (src, trg) in enumerate(batch):
            src_id = [self.token2id[token] for token in src]
            trg_id = [self.sos_id] + [self.token2id[token] for token in trg]
            # print([self.id2token[i] for i in trg_id])
            if len(src_id) < self.src_seq_len:
                src_id = src_id + [0] * (self.src_seq_len - len(src_id))
            else:
                src_id = src_id[:self.src_seq_len]
            if len(trg_id) < self.src_seq_len:
                label_id = trg_id[1:] + [self.eos_id] + [-100] * (self.trg_seq_len - len(trg_id))
                # print([self.id2token[i] for i in trg_id[1:] + [self.eos_id]])
                trg_id = trg_id + [0] * (self.trg_seq_len - len(trg_id))
            else:
                trg_id = trg_id[:self.trg_seq_len]
                label_id = trg_id[1:] + [self.eos_id]
            src_ids.append(src_id)
            trg_ids.append(trg_id)
            label_ids.append(label_id)
        src_ids = torch.tensor(np.array(src_ids)).long()
        trg_ids = torch.tensor(np.array(trg_ids)).long()
        label_ids = torch.tensor(np.array(label_ids)).long()
        data = {
            "src_ids": src_ids,
            "trg_ids": trg_ids,
            "label_ids": label_ids,
        }
        return data


class Args:
    data_path = "./data"
    save_dir = os.path.join("./checkpoints/model.pt")
    src_path = os.path.join(data_path, "duilian/in.txt")
    trg_path = os.path.join(data_path, "duilian/out.txt")
    with open(os.path.join(data_path, "duilian/vocab.txt"), "r", encoding="utf-8") as fp:
        vocab = fp.read().strip().split("\n")
    token2id = {token: i for i, token in enumerate(vocab)}
    id2token = {i: token for i, token in enumerate(vocab)}
    sos_id = 2
    eos_id = 3
    n_src_vocab = len(vocab)
    n_trg_vocab = len(vocab)
    src_pad_idx = 0
    trg_pad_idx = 0
    d_word_vec = 512
    d_model = 512
    d_inner = 1024
    n_layers = 5
    n_head = 8
    d_k = d_model // n_head
    d_v = d_model // n_head
    dropout = 0.1
    src_seq_len = 24
    trg_seq_len = 24
    trg_emb_prj_weight_sharing = False
    emb_src_trg_weight_sharing = False

    epochs = 20
    train_batch_size = 512
    test_batch_size = 5
    learning_rate = 0.000012
    max_grad_norm = 5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1
    do_train = False
    do_predict = True


def predict(model,
            test_dataset,
            args,
            device):
    with torch.no_grad():
        for src, trg in test_dataset:
            if len(src) > args.src_seq_len:
                continue
            trg_id = args.sos_id
            trg_ids_tmp = [[]]
            src_ids = [[args.token2id[token] for token in src]]
            src_ids = torch.tensor(np.array(src_ids)).to(device)
            while trg_id != args.eos_id:
                trg_ids_tmp[0].append(trg_id)
                if len(trg_ids_tmp[0]) == len(src_ids[0]) + 1:
                    break
                trg_ids = torch.tensor(np.array(trg_ids_tmp)).to(device)
                output = model(src_ids, trg_ids)
                output = output.detach().cpu().numpy()
                output = np.argmax(output, -1).tolist()
                trg_id = output[0][-1]

            print("上联：" + src)
            print('真实：' + trg)
            print("下联：" + "".join([args.id2token[i] for i in trg_ids_tmp[0]][1:]))
            print("=" * 100)


def main():
    args = Args()

    data = DLDataset(args)

    ratio = int(0.95 * len(data))
    train_dataset = data[:ratio]
    test_dataset = data[ratio:]
    collate = Collate(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerSeq2Seq(
        n_src_vocab=args.n_src_vocab,
        n_trg_vocab=args.n_trg_vocab,
        src_pad_idx=args.src_pad_idx,
        trg_pad_idx=args.trg_pad_idx,
        d_word_vec=args.d_word_vec,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
        src_seq_len=args.src_seq_len,
        trg_seq_len=args.trg_seq_len,
        trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
        emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing,
    )

    if args.do_train:

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=1,
                                  collate_fn=collate.collate_fn)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=1,
                                 collate_fn=collate.collate_fn)

        if os.path.exists(args.save_dir):
            model.load_state_dict(torch.load(args.save_dir, map_location=torch.device('cpu')), strict=True)

        model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        no_decay = ["bias", "LayerNorm.weight"]
        module = (
            model.module if hasattr(model, "module") else model
        )
        model_param = list(module.named_parameters())

        optimizer_grouped_parameters = [

            {"params": [p for n, p in model_param if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.learning_rate},
            {"params": [p for n, p in model_param if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': args.learning_rate},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        total_step = args.epochs * len(train_loader) + 1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.warmup_proportion * total_step), num_training_steps=total_step
        )

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # print(args.id2token)

        global_step = 1

        for epoch in range(1, args.epochs + 1):
            for step, batch in enumerate(train_loader):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                model.train()
                output = model(
                    batch["src_ids"],
                    batch["trg_ids"],
                )
                batch_size = output.size(0)
                seq_len = output.size(1)
                loss = criterion(output.view(batch_size * seq_len, -1), batch["label_ids"].view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(
                    "epochs:{}/{} steps:{}/{} loss:{:.6f} lr:{:.6f}".format(epoch, args.epochs, global_step, total_step,
                                                                            loss.item(), cur_lr))
                global_step += 1
                if global_step % 100 == 0:
                    predict(model, random.choices(test_dataset, k=10), args, device)
            torch.save(model.state_dict(), args.save_dir)

    if args.do_predict:
        print(args.save_dir)
        model.load_state_dict(torch.load(args.save_dir, map_location=torch.device('cpu')), strict=True)
        model.to(device)
        predict(model, random.choices(test_dataset, k=10), args, device)


if __name__ == '__main__':
    main()
