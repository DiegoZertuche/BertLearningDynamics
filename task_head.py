import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp_mod import SelfAttentiveSpanExtractor
from classifier import Classifier
import numpy as np
import torch_xla.core.xla_model as xm


class EdgeClassifierModule(nn.Module):
    """ Build edge classifier components as a sub-module.
    Use same classifier code as build_single_sentence_module,
    except instead of whole-sentence pooling we'll use span1 and span2 indices
    to extract span representations, and use these as input to the classifier.
    This works in the current form, but with some provisos:
        - Only considers the explicit set of spans in inputs; does not consider
        all other spans as negatives. (So, this won't work for argument
        _identification_ yet.)
    TODO: consider alternate span-pooling operators: max or mean-pooling,
    or SegRNN.
    TODO: add span-expansion to negatives, one of the following modes:
        - all-spans (either span1 or span2), treating not-seen as negative
        - all-tokens (assuming span1 and span2 are length-1), e.g. for
        dependency parsing
        - batch-negative (pairwise among spans seen in batch, where not-seen
        are negative)
    """

    def __init__(self, task_params, device):
        super(EdgeClassifierModule, self).__init__()
        # Set config options needed for forward pass.
        self.span_pooling = task_params["span_pooling"]
        self.single_sided = task_params['single_sided']
        self.proj_dim = task_params["d_hidden"]
        self.n_spans = task_params['n_spans']
        self.n_classes = task_params['n_classes']
        self.loss_type = 'sigmoid'
        self.device = device

        # Span extractor, shared for both span1 and span2.
        if self.device == 'cuda':
            self.span_attention_extractor = SelfAttentiveSpanExtractor(self.proj_dim).to('cuda')
            self.span_attention_extractor_2 = SelfAttentiveSpanExtractor(self.proj_dim).to('cuda')
        elif self.device == 'tpu':
            self.span_attention_extractor = SelfAttentiveSpanExtractor(self.proj_dim).to(xm.xla_device())
            self.span_attention_extractor_2 = SelfAttentiveSpanExtractor(self.proj_dim).to(xm.xla_device())
        else:
            self.span_attention_extractor = SelfAttentiveSpanExtractor(self.proj_dim)
            self.span_attention_extractor_2 = SelfAttentiveSpanExtractor(self.proj_dim)
        self.classifier = Classifier.from_params(self.proj_dim*self.n_spans, self.n_classes, task_params)

    def label_processing(self, labels):
        binary_labels = []
        for label_id in labels:
            binary_label_ids = np.zeros((self.n_classes,), dtype=int)
            binary_label_ids[label_id] = 1
            binary_labels.append(binary_label_ids)
        return torch.tensor(binary_labels)

    def forward(self, batch, sent_mask, predict, device):
        """ Run forward pass.
        Expects batch to have the following entries:
            'batch1' : [batch_size, max_len, ??]
            'labels' : [batch_size, num_targets] of label indices
            'span1s' : [batch_size, num_targets, 2] of spans
            'span2s' : [batch_size, num_targets, 2] of spans
        'labels', 'span1s', and 'span2s' are padded with -1 along second
        (num_targets) dimension.
        Args:
            batch: dict(str -> Tensor) with entries described above.
            sent_mask: [batch_size, max_len, 1] Tensor of {0,1}
            task: EdgeProbingTask
            predict: whether or not to generate predictions
        Returns:
            out: dict(str -> Tensor)
        """
        out = {}

        # Span extraction.
        # [batch_size, num_targets] bool
        span_mask = batch["span1s"][:, :, 0] != -1
        out["mask"] = span_mask
        total_num_targets = span_mask.sum()
        out["n_targets"] = total_num_targets
        out["n_exs"] = total_num_targets  # used by trainer.py

        _kw = dict(sequence_mask=sent_mask.long(), span_indices_mask=span_mask.long())
        # span1_emb and span2_emb are [batch_size, num_targets, span_repr_dim]
        span1_embeddings = self.span_attention_extractor(batch['batch1'], batch["span1s"], device)
        span2_embeddings = self.span_attention_extractor_2(batch['batch1'], batch["span2s"], device)

        span_embeddings = torch.cat([span1_embeddings, span2_embeddings], dim=-1)
        masked_span_embeddings = span_embeddings[span_mask, :]
        masked_labels = batch['labels'][span_mask]

        logits = self.classifier(masked_span_embeddings)
        out["logits"] = logits
        binary_labels = torch.nn.functional.one_hot(masked_labels, self.n_classes)
        out["labels"] = binary_labels

        # Compute loss if requested.
        if "labels" in batch:
            # Labels is [batch_size, num_targets, n_classes],
            # with k-hot encoding provided by AllenNLP's MultiLabelField.
            # Flatten to [total_num_targets, ...] first.
            out["loss"] = self.compute_loss(logits, binary_labels)

        if predict:
            out["preds"] = self.get_predictions(logits)

        return out

    def get_predictions(self, logits: torch.Tensor):
        """Return class probabilities, same shape as logits.
        Args:
            logits: [batch_size, num_targets, n_classes]
        Returns:
            probs: [batch_size, num_targets, n_classes]
        """
        if self.loss_type == "sigmoid":
            return torch.sigmoid(logits)
        else:
            raise ValueError("Unsupported loss type '%s' " "for edge probing." % self.loss_type)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """ Compute loss & eval metrics.
        Expect logits and labels to be already "selected" for good targets,
        i.e. this function does not do any masking internally.
        Args:
            logits: [total_num_targets, n_classes] Tensor of float scores
            labels: [total_num_targets, n_classes] Tensor of sparse binary targets
        Returns:
            loss: scalar Tensor
        """

        if self.loss_type == "sigmoid":
            return F.binary_cross_entropy(torch.sigmoid(logits), labels.float())
        else:
            raise ValueError("Unsupported loss type '%s' " "for edge probing." % self.loss_type)
