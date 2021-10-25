import torch
from torch import nn


class Classifier(nn.Module):
    """ Logistic regression or MLP classifier """

    # NOTE: Expects dropout to have already been applied to its input.

    def __init__(self, d_inp, n_classes, cls_type="mlp", dropout=0.2, d_hid=512):
        super(Classifier, self).__init__()
        if cls_type == "log_reg":
            classifier = nn.Linear(d_inp, n_classes)
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, n_classes),
            )
        elif cls_type == "fancy_mlp":  # What they did in Infersent.
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(p=dropout),
                nn.Linear(d_hid, n_classes),
            )
        else:
            raise ValueError("Classifier type %s not found" % type)
        self.classifier = classifier

    def forward(self, seq_emb):
        logits = self.classifier(seq_emb)
        return logits

    @classmethod
    def from_params(cls, d_inp, n_classes, params):
        return cls(
            d_inp,
            n_classes,
            cls_type=params["cls_type"],
            dropout=params["dropout"],
            d_hid=params["d_hidden"],
        )
