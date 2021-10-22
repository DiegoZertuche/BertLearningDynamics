import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp import SelfAttentiveSpanExtractor
from classifier import Classifier


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

    def __init__(self, task_params):
        super(EdgeClassifierModule, self).__init__()
        # Set config options needed for forward pass.
        self.span_pooling = task_params["span_pooling"]
        self.single_sided = task_params['single_sided']
        self.proj_dim = task_params["d_hidden"]
        self.n_spans = task_params['n_spans']
        self.n_classes = task_params['n_classes']

        # Span extractor, shared for both span1 and span2.
        self.span_attention_extractor = SelfAttentiveSpanExtractor(self.proj_dim)
        self.classifier = Classifier.from_params(self.proj_dim*self.n_spans, self.n_classes)

    def forward(self, batch, sent_mask, task, predict):
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
            word_embs_in_context: [batch_size, max_len, repr_dim] Tensor
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
        span1_emb = self.span_extractors[1](se_proj1, batch["span1s"], **_kw)
        if not self.single_sided:
            span2_emb = self.span_extractors[2](se_proj2, batch["span2s"], **_kw)
            span_emb = torch.cat([span1_emb, span2_emb], dim=2)
        else:
            span_emb = span1_emb

        # [batch_size, num_targets, n_classes]
        logits = self.classifier(span_emb)
        out["logits"] = logits

        # Compute loss if requested.
        if "labels" in batch:
            # Labels is [batch_size, num_targets, n_classes],
            # with k-hot encoding provided by AllenNLP's MultiLabelField.
            # Flatten to [total_num_targets, ...] first.
            out["loss"] = self.compute_loss(logits[span_mask], batch["labels"][span_mask], task)

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

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, task):
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
