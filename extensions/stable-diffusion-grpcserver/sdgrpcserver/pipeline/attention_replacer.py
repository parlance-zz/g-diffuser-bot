# Originally from https://github.com/shunk031/training-free-structured-diffusion-guidance/blob/main/tfsdg/utils/replace_layer.py

import inspect
import torch.nn as nn
from diffusers.models.attention import CrossAttention

def replace_cross_attention(target: nn.Module, crossattention: nn.Module, name: str) -> None:
    for attr_str in dir(target):
        target_attr = getattr(target, attr_str)

        if isinstance(target_attr, CrossAttention):
            query_dim = target_attr.to_q.in_features
            assert target_attr.to_k.in_features == target_attr.to_v.in_features
            context_dim = target_attr.to_k.in_features
            heads = target_attr.heads
            dim_head = int(target_attr.scale**-2)
            dropout = target_attr.to_out[-1].p

            ca_kwargs = {
                "query_dim": query_dim,
                "context_dim": context_dim,
                "heads": heads,
                "dim_head": dim_head,
                "dropout": dropout,
            }

            accepts_struct_attention = "struct_attention" in set(inspect.signature(crossattention.__init__).parameters.keys())

            if accepts_struct_attention:
                ca_kwargs["struct_attention"] = (attr_str == "attn2")

            ca = crossattention(**ca_kwargs)

            original_params = list(target_attr.parameters())
            proposed_params = list(ca.parameters())
            assert len(original_params) == len(proposed_params)

            for p1, p2 in zip(original_params, proposed_params):
                p2.data.copy_(p1.data)

            setattr(target, attr_str, ca)

    for name, immediate_child_module in target.named_children():
        replace_cross_attention(target=immediate_child_module, crossattention=crossattention, name=name)