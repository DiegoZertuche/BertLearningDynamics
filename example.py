from main_model import MainModel
import torch
from torch.utils.data import DataLoader
from preprocess import MyDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from allennlp.training.metrics import F1MultiLabelMeasure
import numpy as np

params = {
    'input_module': 'google/multiberts-seed-4-1900k',
    'transfer_mode': 'frozen',
    'output_mode': 'mix',
    'max_layer': -1
}

task_params = {
    'span_pooling': 'attn',
    'n_spans': 2,
    'n_classes': 49,
    'single_sided': False,
    'cls_type': 'mlp',
    'dropout': 0.2
}

model = MainModel(params, task_params)


def update_metrics(out, f1_scorer):
    preds = out["preds"]
    labels = out["labels"]
    f1_scorer(preds, labels)


def tokenize_data(texts):
    return model.encoder.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True)


def pad_collate(batch):
    (xx, spans1, spans2, labels) = zip(*batch)
    xx_pad = tokenize_data(xx)
    spans1_padded = pad_sequence(spans1, batch_first=True, padding_value=-1)
    spans2_padded = pad_sequence(spans2, batch_first=True, padding_value=-1)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return xx_pad, spans1_padded, spans2_padded, labels_padded


train_task_filepath = 'probing/data/edges/dep_ewt/en_ewt-ud-train.json'
labels_filepath = 'probing/data/edges/dep_ewt/labels.txt'
train_ds = MyDataset(train_task_filepath, train_task_filepath + '.retokenized.bert-base-uncased', labels_filepath)
dev_task_filepath = 'probing/data/edges/dep_ewt/en_ewt-ud-dev.json'
dev_ds = MyDataset(dev_task_filepath, dev_task_filepath + '.retokenized.bert-base-uncased', labels_filepath)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=pad_collate)
dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=pad_collate)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.001)
f1_scorer = F1MultiLabelMeasure(average="micro")

train_losses = []
test_losses = []
train_f1 = []
val_f1 = []
train_acc = []
val_acc = []

n_epochs = 10
for epoch in range(n_epochs):
    running_loss = 0
    f1_scorer.reset()
    for batch in tqdm(train_dl, leave=False):
        outs = model(batch, True)
        outs['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += outs['loss'].cpu().item()
        update_metrics(outs, f1_scorer)

    train_losses.append(running_loss)
    print("=" * 20)
    print(f"Epoch {epoch + 1}/{n_epochs} Train Loss: {running_loss}")
    print(f"Epoch {epoch + 1}/{n_epochs} Train F1: {f1_scorer.get_metric()}")

    running_loss = 0
    f1_scorer.reset()
    with torch.no_grad():
        for batch in tqdm(dev_dl, leave=False):
            outs = model(batch, True)
            running_loss += outs['loss'].cpu().item()
            update_metrics(outs, f1_scorer)

    print(f"Epoch {epoch + 1}/{n_epochs} Val Loss: {running_loss}")
    print(f"Epoch {epoch + 1}/{n_epochs} Val Metrics: {f1_scorer.get_metric()}")

#write_pickle(f'{PARENT_DIR}model-w2v.pk', model)
print('DONE')
