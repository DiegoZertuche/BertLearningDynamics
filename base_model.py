import transformers
import torch
import torch.nn as nn
from typing import Dict
from allennlp_mod import ScalarMix


class BaseModel(nn.Module):
    """ Shared code for transformers wrappers.
    Subclasses share a good deal of code, but have a number of subtle differences due to different
    APIs from transfromers.
    """

    def __init__(self, params):
        super().__init__()
        self.input_module = params['input_module']
        self.transfer_mode = params['transfer_mode']
        self.output_mode = params['output_mode']
        self.model = transformers.BertModel.from_pretrained(self.input_module, output_hidden_states=True)
        self.max_pos = self.model.config.max_position_embeddings
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.input_module)

        # Integer token indices for special symbols.
        self._sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self._cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self._pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self._unk_id = self.tokenizer.convert_tokens_to_ids("[UNK]")
        self._mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")

        self.parameter_setup(self.transfer_mode, params['max_layer'], self.output_mode)

    def parameter_setup(self, transfer_mode, max_layer, output_mode):
        # Set trainability of this module.
        for param in self.model.parameters():
            param.requires_grad = bool(transfer_mode == "finetune")

        self.num_layers = self.model.config.num_hidden_layers
        if max_layer >= 0:
            self.max_layer = max_layer
            assert self.max_layer <= self.num_layers
        else:
            self.max_layer = self.num_layers

        # Configure scalar mixing, ELMo-style.
        if output_mode == "mix":
            self.scalar_mix = ScalarMix(self.max_layer + 1, do_layer_norm=False)

    def correct_sent_indexing(self, sent):
        """ Correct id difference between transformers and AllenNLP.
        The AllenNLP indexer adds'@@UNKNOWN@@' token as index 1, and '@@PADDING@@' as index 0
        args:
            sent: batch dictionary, in which
                sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs
        returns:
            ids: <long> [bath_size, var_seq_len] corrected token IDs
            input_mask: <long> [bath_size, var_seq_len] mask of input sequence
        """
        assert (
            self.tokenizer_required in sent
        ), "transformers cannot find correcpondingly tokenized input"
        ids = sent[self.tokenizer_required]

        input_mask = (ids != 0).long()
        pad_mask = (ids == 0).long()
        # map AllenNLP @@PADDING@@ to _pad_id in specific transformer vocab
        unk_mask = (ids == 1).long()
        # map AllenNLP @@UNKNOWN@@ to _unk_id in specific transformer vocab
        valid_mask = (ids > 1).long()
        # shift ordinary indexes by 2 to match pretrained token embedding indexes
        if self._unk_id is not None:
            ids = (ids - 2) * valid_mask + self._pad_id * pad_mask + self._unk_id * unk_mask
        else:
            ids = (ids - 2) * valid_mask + self._pad_id * pad_mask
            assert (
                unk_mask == 0
            ).all(), "out-of-vocabulary token found in the input, but _unk_id of transformers model is not specified"
        if self.max_pos is not None:
            assert (
                ids.size()[-1] <= self.max_pos
            ), "input length exceeds position embedding capacity, reduce max_seq_len"

        sent[self.tokenizer_required] = ids
        return ids, input_mask

    def prepare_output(self, lex_seq, hidden_states, input_mask):
        """
        Convert the output of the transformers module to a vector sequence as expected by jiant.
        args:
            lex_seq: The sequence of input word embeddings as a tensor (batch_size, sequence_length, hidden_size).
                     Used only if output_mode = "only".
            hidden_states: A list of sequences of model hidden states as tensors (batch_size, sequence_length, hidden_size).
            input_mask: A tensor with 1s in positions corresponding to non-padding tokens (batch_size, sequence_length).
        returns:
            h: Output embedding as a tensor (batch_size, sequence_length, output_dim)
        """
        available_layers = hidden_states[: self.max_layer + 1]

        if self.output_mode in ["none", "top"]:
            h = available_layers[-1]
        elif self.output_mode == "only":
            h = lex_seq
        elif self.output_mode == "cat":
            h = torch.cat([available_layers[-1], lex_seq], dim=2)
        elif self.output_mode == "mix":
            h = self.scalar_mix(available_layers, mask=input_mask)
        else:
            raise NotImplementedError(f"output_mode={self.output_mode}" " not supported.")

        return h

    def get_output_dim(self):
        if self.output_mode == "cat":
            return 2 * self.model.config.hidden_size
        else:
            return self.model.config.hidden_size

    def get_seg_ids(self, token_ids, input_mask):
        """ Dynamically build the segment IDs for a concatenated pair of sentences
        Searches for index _sep_id in the tensor. Supports BERT or XLNet-style padding.
        Sets padding tokens to segment zero.
        args:
            token_ids (torch.LongTensor): batch of token IDs
            input_mask (torch.LongTensor): mask of token_ids
        returns:
            seg_ids (torch.LongTensor): batch of segment IDs
        example:
        > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]", "[PAD]"]
        > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
        > seg_ids = get_seg_ids(token_tensor, torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]))
        > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        """
        # TODO: creating sentence segment id(and language segment id for XLM) is more suitable for preprocess
        sep_idxs = (token_ids == self._sep_id).long()
        sep_count = torch.cumsum(sep_idxs, dim=-1) - sep_idxs
        seg_ids = sep_count * input_mask

        return seg_ids

    @staticmethod
    def apply_boundary_tokens(s1, s2=None, get_offset=False):
        # BERT-style boundary token padding on string token sequences
        if s2:
            s = ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
            if get_offset:
                return s, 1, len(s1) + 2
        else:
            s = ["[CLS]"] + s1 + ["[SEP]"]
            if get_offset:
                return s, 1
        return s

    def forward(self, sent): #TODO: MODIFY
        #ids, input_mask = self.correct_sent_indexing(sent)
        hidden_states, lex_seq = [], None
        #if self.output_mode not in ["none", "top"]:
        #    lex_seq = self.model.embeddings.word_embeddings(ids)
        #    lex_seq = self.model.embeddings.LayerNorm(lex_seq)
        if self.output_mode != "only":
            #oken_types = self.get_seg_ids(ids, input_mask)
            outs = self.model(
                input_ids=sent['input_ids'],
                token_type_ids=sent['token_type_ids'],
                attention_mask=sent['attention_mask'])

        return self.prepare_output(lex_seq, outs.hidden_states, sent['attention_mask'])

    def get_pretrained_lm_head(self):
        model_with_lm_head = transformers.BertForMaskedLM.from_pretrained(
            self.input_module, cache_dir=self.cache_dir
        )
        lm_head = model_with_lm_head.cls
        lm_head.predictions.decoder.weight = self.model.embeddings.word_embeddings.weight
        return lm_head
