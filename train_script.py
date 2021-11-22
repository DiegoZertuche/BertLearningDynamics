from main_model import MainModel
import torch
from torch.utils.data import DataLoader
from preprocess import MyDataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from tqdm import tqdm
from allennlp.training.metrics import F1MultiLabelMeasure
import numpy as np
import pickle
import torch_xla.core.xla_model as xm


def update_metrics(preds, labels, f1_scorer):
    f1_scorer(preds, labels)


def tokenize_data(texts):
    global model_
    return model_.encoder.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True)


def pad_collate(batch):
    (xx, spans1, spans2, labels) = zip(*batch)
    xx_pad = tokenize_data(xx)
    spans1_padded = pad_sequence(spans1, batch_first=True, padding_value=-1)
    spans2_padded = pad_sequence(spans2, batch_first=True, padding_value=-1)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return xx_pad, spans1_padded, spans2_padded, labels_padded

def log_and_save(losses, metrics, filename):
  with open(filename+'-losses.txt', 'w') as f:
    for loss in losses:
      f.write('{}'.format(loss))
      f.write('\n')
    f.close()

  with open(filename+'-metrics.txt', 'w') as f:
    for metric in metrics:
      f.write('{} {} {}'.format(metric['precision'], metric['recall'], metric['fscore']))
      f.write('\n')
    f.close()

def fetch_and_tokenize_data(params, task_params, device, train_task_filepath, labels_filepath, dev_task_filepath, n_spans=2):
    model = MainModel(params, task_params, device='tpu')
    train_ds = MyDataset(train_task_filepath, train_task_filepath + '.retokenized.bert-base-uncased', labels_filepath, n_spans=n_spans)
    dev_ds = MyDataset(dev_task_filepath, dev_task_filepath + '.retokenized.bert-base-uncased', labels_filepath, n_spans=n_spans)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=pad_collate)
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=pad_collate)

    return train_ds, dev_ds, train_dl, dev_dl

def run_batch(model_strings, filenames, epochs, dev, params, task_params, train_dl, dev_dl):
  for model_string, filename in zip(model_strings, filenames):
    print('='*30)
    model = MainModel(params, task_params, device=dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    f1_scorer = F1MultiLabelMeasure(average="micro")

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    n_epochs = epochs
    print('Training model {}'.format(model_string))
    for epoch in range(n_epochs):
        running_loss = 0
        f1_scorer.reset()
        for batch in tqdm(train_dl, leave=False):
            outs = model(batch, True)
            outs['loss'].backward()
            #optimizer.step()
            xm.optimizer_step(optimizer)
            print('xm')
            #running_loss += outs['loss'].cpu()
            print('dooooo')
            #preds = outs['preds'].cpu()
            #print(preds)
            print('ne')
            #labels = outs['labels'].cpu()
            print('boi')
            #f1_scorer(preds, labels)            
            print('update')
            optimizer.zero_grad()

        train_losses.append(running_loss)
        train_metrics.append(f1_scorer.get_metric())
        print("=" * 20)
        print(f"Epoch {epoch + 1}/{n_epochs} Train Loss: {running_loss}")
        print(f"Epoch {epoch + 1}/{n_epochs} Train Metrics: {f1_scorer.get_metric()}")

        running_loss = 0
        f1_scorer.reset()
        with torch.no_grad():
            for batch in tqdm(dev_dl, leave=False):
                outs = model(batch, True)
                running_loss += outs['loss'].cpu().item()
                update_metrics(outs, f1_scorer)

        val_losses.append(running_loss)
        val_metrics.append(f1_scorer.get_metric())
        print(f"Epoch {epoch + 1}/{n_epochs} Val Loss: {running_loss}")
        print(f"Epoch {epoch + 1}/{n_epochs} Val Metrics: {f1_scorer.get_metric()}")
    torch.save(model.state_dict(), '/content/gdrive/MyDrive/Google_2021_Capstone/dev/results/ud/seed-0/{}-dict.pth'.format(filename))
    log_and_save(train_losses, train_metrics, '/content/gdrive/MyDrive/Google_2021_Capstone/dev/results/ud/seed-0/{}-train'.format(filename))
    log_and_save(val_losses, val_metrics, '/content/gdrive/MyDrive/Google_2021_Capstone/dev/results/ud/seed-0/{}-val'.format(filename))


if __name__ == '__main__':
    params = {
		    'input_module': 'MultiBertGunjanPatrick/multiberts-seed-4',
		    'transfer_mode': 'frozen',
		    'output_mode': 'mix',
		    'max_layer': -1
		}

    task_params = {
		    'span_pooling': 'attn',
		    'n_spans': 2,
		    'n_classes': 19,
		    'single_sided': False,
		    'cls_type': 'mlp',
		    'dropout': 0.2
	}

    train_task_filepath = 'edges/semeval/train.0.85.json'
    labels_filepath = 'edges/semeval/labels.txt'
    dev_task_filepath = 'edges/semeval/dev.json'

    model_ = MainModel(params, task_params, device='tpu')
    train_ds, dev_ds, train_dl, dev_dl = fetch_and_tokenize_data(params, task_params, 'tpu', train_task_filepath, 
		labels_filepath, dev_task_filepath)

    multibert_string = "MultiBertGunjanPatrick/multiberts-seed-"
    seeds = [str(x) for x in range(25)]
    checkpoints = ['0k', '20k', '60k', '100k', '200k', '400k', '700k', '1000k', 
               	   '1500k', '1800k', '2000k']

    model_list = [multibert_string+seeds[0] + '-' + check for check in checkpoints]
    filenames = checkpoints

    run_batch(model_list, filenames, 20, 'tpu', params, task_params, train_dl, dev_dl)

