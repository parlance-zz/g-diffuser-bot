import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Literal

import numpy as np
import stanza
from stanza.pipeline.core import DownloadMethod
import torch
from nltk.tree import Tree
from transformers.tokenization_utils import BatchEncoding

STRUCT_ATTENTION_TYPE = Literal["extend_str", "extend_seq", "align_seq", "none"]

@dataclass
class KeyValueTensors(object):
    k: torch.Tensor
    v: torch.Tensor

@dataclass
class Span(object):
    left: int
    right: int


@dataclass
class SubNP(object):
    text: str
    span: Span


@dataclass
class AllNPs(object):
    nps: List[str]
    spans: List[Span]
    lowest_nps: List[SubNP]


stanza_singleton = None

def get_stanza_singleton():
    global stanza_singleton

    if stanza_singleton is None:
        stanza_singleton = stanza.Pipeline(
            lang="en", processors="tokenize,pos,constituency", use_gpu=False, download_method=DownloadMethod.REUSE_RESOURCES
        )

    return stanza_singleton



class StructuredTextEmbedding:
    def __init__(self, pipe, struct_attention: STRUCT_ATTENTION_TYPE = "none",):
        self.pipe = pipe

        self.nlp = get_stanza_singleton()

        self.struct_attention = struct_attention

    def preprocess_prompt(self, prompt: str) -> str:
        return prompt.lower().strip().strip(".").strip()

    def get_sub_nps(self, tree: Tree, left: int, right: int) -> List[SubNP]:

        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []

        sub_nps: List[SubNP] = []

        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[: len(n_subtree_leaves)]
        assert right - left == n_leaves

        if tree.label() == "NP" and n_leaves > 1:
            sub_np = SubNP(
                text=" ".join(tree.leaves()),
                span=Span(left=int(left), right=int(right)),
            )
            sub_nps.append(sub_np)

        for i, subtree in enumerate(tree):
            sub_nps += self.get_sub_nps(
                subtree,
                left=left + offset[i],
                right=left + offset[i] + n_subtree_leaves[i],
            )
        return sub_nps

    def get_all_nps(self, tree: Tree, full_sent: Optional[str] = None) -> AllNPs:
        start = 0
        end = len(tree.leaves())

        all_sub_nps = self.get_sub_nps(tree, left=start, right=end)

        lowest_nps = []
        for i in range(len(all_sub_nps)):
            span = all_sub_nps[i].span
            lowest = True
            for j in range(len(all_sub_nps)):
                span2 = all_sub_nps[j].span
                if span2.left >= span.left and span2.right <= span.right:
                    lowest = False
                    break
            if lowest:
                lowest_nps.append(all_sub_nps[i])

        all_nps = [all_sub_np.text for all_sub_np in all_sub_nps]
        spans = [all_sub_np.span for all_sub_np in all_sub_nps]

        if full_sent and full_sent not in all_nps:
            all_nps = [full_sent] + all_nps
            spans = [Span(left=start, right=end)] + spans

        return AllNPs(nps=all_nps, spans=spans, lowest_nps=lowest_nps)

    def tokenize(self, prompt: Union[str, List[str]]) -> BatchEncoding:
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_input

    def _extend_string(self, nps: List[str]) -> List[str]:
        extend_nps: List[str] = []
        for i in range(len(nps)):
            if i == 0:
                extend_nps.append(nps[i])
            else:
                np = (" " + nps[i]) * (
                    self.pipe.tokenizer.model_max_length // len(nps[i].split())
                )
                extend_nps.append(np)
        return extend_nps

    def _expand_sequence(
        self, seq: torch.Tensor, length: int, dim: int = 1
    ) -> torch.Tensor:

        # shape: (77, 768) -> (768, 77)
        seq = seq.transpose(0, dim)

        max_length = seq.size(0)
        n_repeat = (max_length - 2) // length

        # shape: (10, 1)
        repeat_size = (n_repeat,) + (1,) * (len(seq.size()) - 1)

        # shape: (77,)
        eos = seq[length + 1, ...].clone()

        # shape: (750, 77)
        segment = seq[1 : length + 1, ...].repeat(*repeat_size)

        seq[1 : len(segment) + 1] = segment

        # To avoid the following error, we need to use `torch.no_grad` function:
        # RuntimeError: Output 0 of SliceBackward0 is a view and
        # # is being modified inplace. This view is the output
        # of a function that returns multiple views.
        # Such functions do not allow the output views to be modified inplace.
        # You should replace the inplace operation by an out-of-place one.
        seq[len(segment) + 1] = eos

        # shape: (768, 77) -> (77, 768)
        return seq.transpose(0, dim)

    def _align_sequence(
        self,
        full_seq: torch.Tensor,
        seq: torch.Tensor,
        span: Span,
        eos_loc: int,
        dim: int = 1,
        zero_out: bool = False,
        replace_pad: bool = False,
    ) -> torch.Tensor:

        # shape: (77, 768) -> (768, 77)
        seq = seq.transpose(0, dim)

        # shape: (77, 768) -> (768, 77)
        full_seq = full_seq.transpose(0, dim)

        start, end = span.left + 1, span.right + 1
        seg_length = end - start

        full_seq[start:end] = seq[1 : 1 + seg_length]
        if zero_out:
            full_seq[1:start] = 0
            full_seq[end:eos_loc] = 0

        if replace_pad:
            pad_length = len(full_seq) - eos_loc
            full_seq[eos_loc:] = seq[1 + seg_length : 1 + seg_length + pad_length]

        # shape: (768, 77) -> (77, 768)
        return full_seq.transpose(0, dim)

    def extend_str(self, nps: List[str]) -> torch.Tensor:
        nps = self._extend_string(nps)

        input_ids = self.tokenize(nps).input_ids
        enc_output = self.pipe.text_encoder(input_ids.to(self.pipe.device))
        c = enc_output.last_hidden_state
        return c

    def extend_seq(self, nps: List[str]):

        input_ids = self.tokenize(nps).input_ids

        # repeat each NP after embedding
        nps_length = [len(ids) - 2 for ids in input_ids]  # not including bos eos

        enc_output = self.pipe.text_encoder(input_ids.to(self.pipe.device))
        c = enc_output.last_hidden_state

        # shape: (num_nps, model_max_length, hidden_dim)
        c = torch.stack(
            [c[0]]
            + [self._expand_sequence(seq, l) for seq, l in zip(c[1:], nps_length[1:])]
        )
        return c

    def align_seq(self, nps: List[str], spans: List[Span]) -> KeyValueTensors:

        input_ids = self.tokenize(nps).input_ids
        nps_length = [len(ids) - 2 for ids in input_ids]
        enc_output = self.pipe.text_encoder(input_ids.to(self.pipe.device))
        c = enc_output.last_hidden_state

        # shape: (num_nps, model_max_length, hidden_dim)
        k_c = torch.stack(
            [c[0]]
            + [
                self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
                for seq, span in zip(c[1:], spans[1:])
            ]
        )
        # shape: (num_nps, model_max_length, hidden_dim)
        v_c = torch.stack(
            [c[0]]
            + [
                self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
                for seq, span in zip(c[1:], spans[1:])
            ]
        )
        return KeyValueTensors(k=k_c, v=v_c)

    def apply_text_encoder(
        self,
        struct_attention: STRUCT_ATTENTION_TYPE,
        prompt: str,
        nps: List[str],
        spans: Optional[List[Span]] = None,
    ) -> Union[torch.Tensor, KeyValueTensors]:

        if struct_attention == "extend_str":
            return self.extend_str(nps=nps)

        elif struct_attention == "extend_seq":
            return self.extend_seq(nps=nps)

        elif struct_attention == "align_seq" and spans is not None:
            return self.align_seq(nps=nps, spans=spans)

        elif struct_attention == "none":
            text_input = self.tokenize(prompt)
            return self.pipe.text_encoder(text_input.input_ids.to(self.pipe.device))[0]

        else:
            raise ValueError(f"Invalid type of struct attention: {struct_attention}")

    def get_text_embeddings(self, prompt, uncond_prompt):
        preprocessed_prompt = self.preprocess_prompt(prompt.as_unweighted_string()[0])

        doc = self.nlp(preprocessed_prompt)
        tree = Tree.fromstring(str(doc.sentences[0].constituency))

        all_nps = self.get_all_nps(tree=tree, full_sent=preprocessed_prompt)

        text_embeddings = self.apply_text_encoder(
            struct_attention=self.struct_attention,
            prompt=preprocessed_prompt,
            nps=all_nps.nps,
            spans=all_nps.spans,
        )

        return text_embeddings, None
