import torch
from torch import nn
import math


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, linear=False, num_heads=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model: int = d_model
        self.d_k: int = self.d_model // num_heads
        self.sqrt_d_k: float = math.sqrt(self.d_k)
        self.num_heads = num_heads
        self.attns = None
        self.linear = linear

        if self.linear:
            self.proj_q = nn.Linear(self.d_model, self.d_k)
            self.proj_k = nn.Linear(self.d_model, self.d_k)
            self.proj_v = nn.Linear(self.d_model, self.d_k)
            self.proj_o = nn.Linear(num_heads * self.d_k, self.d_model)

        self.softmax = nn.Softmax(dim=-1)

        # TODO: The original implementation appears to not
        # even have a correct implementation for multiple heads
        # (maybe they only use 1 head because they couldn't get
        # multihead working.) Consider implementing MHA properly,
        # with one set of linear projections per head.

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # skip linear, not used in final implementation
        # also skip scaling, apparently, according to the paper.
        attn_weights: torch.Tensor = self.softmax(query @ key.mT)
        attn_values = torch.reshape(
            input=attn_weights @ value,
            shape=(
                attn_weights.shape[0],
                -1,
                self.d_model
            )
        )
        return attn_values, attn_weights
