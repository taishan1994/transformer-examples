import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from model import TransformerEncoder


class CLSDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as fp:
            srcs = fp.read().strip().split("\n")
        for src in srcs:
            src = src.split("\t")
            label = src[0]
            text = src[1]
            self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class Collate:
    def __init__(self, args):
        self.src_seq_len = args.src_seq_len
        self.token2id = args.token2id
        self.id2token = args.id2token
        self.label2id = args.label2id

    def collate_fn(self, batch):
        src_ids = []
        label_ids = []
        for i, (text, label) in enumerate(batch):
            src_id = [self.token2id.get(token, 1) for token in text]
            if len(src_id) < self.src_seq_len:
                src_id = src_id + [0] * (self.src_seq_len - len(src_id))
            else:
                src_id = src_id[:self.src_seq_len]
            label_ids.append(self.label2id[label])
            src_ids.append(src_id)
        src_ids = torch.tensor(np.array(src_ids)).long()
        label_ids = torch.tensor(np.array(label_ids)).long()
        data = {
            "src_ids": src_ids,
            "label_ids": label_ids,
        }
        return data


class Args:
    data_path = "./data"
    save_dir = os.path.join("./checkpoints/ae_model.pt")
    train_path = os.path.join(data_path, "cnews/cnews.train.txt")
    test_path = os.path.join(data_path, "cnews/cnews.test.txt")
    dev_path = os.path.join(data_path, "cnews/cnews.val.txt")
    with open(os.path.join(data_path, "cnews/cnews.vocab.txt"), "r", encoding="utf-8") as fp:
        vocab = fp.read().strip().split("\n")
    token2id = {token: i for i, token in enumerate(vocab)}
    id2token = {i: token for i, token in enumerate(vocab)}
    labels = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    label_num = len(labels)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    n_src_vocab = len(vocab)
    src_pad_idx = 0
    d_word_vec = 512
    d_model = 512
    d_inner = 1024
    n_layers = 5
    n_head = 8
    d_k = d_model // n_head
    d_v = d_model // n_head
    dropout = 0.1
    src_seq_len = 512

    epochs = 20
    train_batch_size = 64
    test_batch_size = 5
    learning_rate = 0.000012
    max_grad_norm = 5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1
    do_train = False
    do_predict = True
    do_test = False


def predict(model,
            test_dataset,
            args,
            device):
    with torch.no_grad():
        for text, label in test_dataset:
            if len(text) > args.src_seq_len:
                text = text[:args.src_seq_len]
            src_ids = [[args.token2id.get(token, 1) for token in text]]
            src_ids = torch.tensor(np.array(src_ids)).to(device)
            output = model(src_ids)
            output = output.detach().cpu().numpy()
            output = np.argmax(output, -1).tolist()

            print("文本：" + text[:100] + "......")
            print('真实标签：' + label)
            print("预测标签：" + args.id2label[output])
            print("=" * 100)


def test(model, test_loader, args, device):
    with torch.no_grad():
        trues = []
        preds = []
        for step, batch in enumerate(test_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            output = model(
                batch["src_ids"],
            )
            trues.extend(batch["label_ids"].detach().cpu().numpy().tolist())
            output = np.argmax(output.detach().cpu().numpy(), -1).tolist()
            preds.extend(output)
    assert len(preds) == len(trues)
    print(classification_report(trues, preds, target_names=args.labels))


def main():
    args = Args()

    train_dataset = CLSDataset(args.train_path)
    test_dataset = CLSDataset(args.test_path)
    dev_dataset = CLSDataset(args.dev_path)

    collate = Collate(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerEncoder(
        n_src_vocab=args.n_src_vocab,
        src_pad_idx=args.src_pad_idx,
        d_word_vec=args.d_word_vec,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
        src_seq_len=args.src_seq_len,
        label_num=args.label_num,
    )

    if args.do_train:

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=1,
                                  collate_fn=collate.collate_fn)
        dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=1,
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
                )
                loss = criterion(output, batch["label_ids"].view(-1))
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
                    predict(model, random.choices(test_dataset[:10000], k=10), args, device)
            torch.save(model.state_dict(), args.save_dir)

    if args.do_predict:
        print(args.save_dir)
        model.load_state_dict(torch.load(args.save_dir, map_location=torch.device('cpu')), strict=True)
        model.to(device)
        predict(model, random.choices(test_dataset, k=10), args, device)

    if args.do_test:
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=1,
                                 collate_fn=collate.collate_fn)
        model.load_state_dict(torch.load(args.save_dir, map_location=torch.device('cpu')), strict=True)
        model.to(device)
        test(model, test_loader, args, device)


if __name__ == '__main__':
    main()
