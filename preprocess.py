import os
import torch
import pandas as pd
from functools import partial
import numpy as np
import boto3
import awswrangler as wr


def get_map_label_to_idx(file_name):
    mapper = {}
    s3_client = boto3.client(service_name='s3')
    # get S3 object
    result = s3_client.get_object(Bucket='multiberts-capstone-2021', Key=file_name) 
    #Read a text file line by line using splitlines object
    for i, line in enumerate(result["Body"].read().splitlines()):
        each_line = line.decode('utf-8')
        mapper[each_line.strip('\n')] = i

    return mapper


def realign_spans(record):
    """
    Builds the indices alignment while also tokenizing the input
    piece by piece.
    Currently, SentencePiece (for XLNet), WPM (for BERT), BPE (for GPT/XLM),
    ByteBPE (for RoBERTa/GPT-2) and Moses (for Transformer-XL and default) tokenization are
    supported.
    Parameters
    -----------------------
        record: dict with the below fields
            text: str
            targets: list of dictionaries
                label: bool
                span1_index: int, start index of first span
                span1_text: str, text of first span
                span2_index: int, start index of second span
                span2_text: str, text of second span
        tokenizer_name: str
    Returns
    ------------------------
        record: dict with the below fields:
            text: str in tokenized form
            targets: dictionary with the below fields
                -label: bool
                -span_1: (int, int) of token indices
                -span1_text: str, the string
                -span2: (int, int) of token indices
                -span2_text: str, the string
    """
    for i, target in enumerate(record):
        span1 = target["span1"]
        span2 = target["span2"]

        # align spans and make them end inclusive
        span1 = [span1[0]+1, span1[1]]
        span2 = [span2[0]+1, span2[1]]

        record[i]['span1'] = span1
        record[i]['span2'] = span2

    return record


def load_span_data(file_name, file_name_retokenized, label_fn=None, has_labels=True):
    """
    Load a span-related task file in .jsonl format, does re-alignment of spans, and tokenizes
    the text.
    Re-alignment of spans involves transforming the spans so that it matches the text after
    tokenization.
    For example, given the original text: [Mr., Porter, is, nice] and bert-base-cased
    tokenization, we get [Mr, ., Por, ter, is, nice ]. If the original span indices was [0,2],
    under the new tokenization, it becomes [0, 3].
    The task file should of be of the following form:
        text: str,
        label: bool
        target: dict that contains the spans
    Args:
        tokenizer_name: str,
        file_name: str,
        label_fn: function that expects a row and outputs a transformed row with labels
          transformed.
    Returns:
        List of dictionaries of the aligned spans and tokenized text.
    """
    
    retokenized_rows = wr.s3.read_json(path=[file_name_retokenized], lines=True)
    rows = wr.s3.read_json(path=[file_name], lines=True)
    bools = rows['targets'].apply(lambda x: len(x) != 0)
    rows = rows[bools]
    retokenized_rows = retokenized_rows[bools]
    # realign spans
    retokenized_rows['targets'] = retokenized_rows['targets'].apply(lambda x: realign_spans(x))
    if has_labels is False:
        retokenized_rows["label"] = 0
    elif label_fn is not None:
        retokenized_rows["label"] = retokenized_rows["label"].apply(label_fn)

    retokenized_rows["text"] = rows["text"]
    return list(retokenized_rows.T.to_dict().values())


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filename, filename_retokenized, filename_labels, label_fn=None):
        self.filename = filename
        self.filename_retokenized = filename_retokenized
        self.rows = load_span_data(self.filename, self.filename_retokenized, label_fn)
        self.texts = [x['text'] for x in self.rows]
        self.span1s = [[info['span1'] for info in x['targets']] for x in self.rows]
        self.span2s = [[info['span2'] for info in x['targets']] for x in self.rows]
        self.mapper = get_map_label_to_idx(filename_labels)
        print(self.mapper)
        self.label_num = len(self.mapper)
        self.labels = [[self.mapper[info['label']] for info in x['targets']] for x in self.rows]
            

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.span1s[idx]), torch.tensor(self.span2s[idx]), torch.tensor(self.labels[idx])
