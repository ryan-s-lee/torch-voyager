import torch
import sys
from torch import nn

from mha import MultiHeadedAttention


class Prefetcher(nn.Module):
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        pc_vocab_size: int,
        page_vocab_size: int,
        page_out_vocab_size: int,
        pc_embed_size: int,
        page_embed_size: int,
        num_experts: int = 100,
        lstm_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # Initialize model members
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.offset_size = 512
        self.pc_embed_size = pc_embed_size
        self.page_embed_size = page_embed_size
        self.num_experts = num_experts
        self.offset_embed_size = page_embed_size * num_experts
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.pc_vocab_size = pc_vocab_size
        self.page_vocab_size = page_vocab_size
        self.page_out_vocab_size = page_out_vocab_size
        self.dropout = dropout

        # Embed Layer
        # TODO: Consider initializing with different weights;
        # see original paper, which at a glance seems to limit range to -1,1
        # rather than Gaussian.
        self.pc_embed = nn.Embedding(
            num_embeddings=self.pc_vocab_size, embedding_dim=self.pc_embed_size
        )
        self.page_embed = nn.Embedding(
            num_embeddings=self.page_vocab_size, embedding_dim=self.page_embed_size
        )
        self.offset_embed = nn.Embedding(
            num_embeddings=self.offset_size, embedding_dim=self.offset_embed_size
        )

        # MHA Layer. Because the multi-expert design
        # requires 4-d inputs, we cannot use PyTorch's default MHA.
        self.pa_off_embed = MultiHeadedAttention(d_model=self.page_embed_size)
        # LSTM Layer
        self.page_lstm = nn.LSTM(
            input_size=self.pc_embed_size
            + self.page_embed_size
            + self.page_embed_size,
            hidden_size=lstm_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.offset_lstm = nn.LSTM(
            input_size=self.pc_embed_size
            + self.page_embed_size
            + self.page_embed_size,  # pa_offset will be the same size as page offset
            hidden_size=lstm_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.page_logits = nn.Linear(self.lstm_size, self.page_out_vocab_size)
        self.offset_logits = nn.Linear(self.lstm_size, self.offset_size)

    def forward(
        self, pc_seq: torch.Tensor, page_seq: torch.Tensor, off_seq: torch.Tensor
    ):
        # obtain pc, page, and offset embeddings
        # print(pc_seq, file=sys.stderr)
        pc_embed_out = self.pc_embed(pc_seq)
        page_embed_out = self.page_embed(page_seq)
        off_embed_out = self.offset_embed(off_seq)

        # obtain page-aware offset embedding
        pa_key_value = torch.reshape(
            input=off_embed_out,
            shape=(
                off_seq.shape[0],
                self.seq_len,
                self.num_experts,
                self.page_embed_size,
            ),
        )
        pa_query = torch.reshape(
            input=page_embed_out,
            shape=(
                page_seq.shape[0],
                self.seq_len,
                1,
                self.page_embed_size,
            ),
        )
        pa_off_embed_out, _ = self.pa_off_embed(pa_query, pa_key_value, pa_key_value)

        # concatenate inputs and pass them to lstms
        # print(pc_embed_out.shape, page_embed_out.shape, pa_off_embed_out.shape)
        lstm_input = torch.concat(
            tensors=(pc_embed_out, page_embed_out, pa_off_embed_out), dim=-1
        )

        page_lstm_out, _ = self.page_lstm(lstm_input)
        offset_lstm_out, _ = self.offset_lstm(lstm_input)

        # pass lstm results through fully connected layer
        # to obtain embedding
        page_out_embed = self.page_logits(page_lstm_out)
        offset_out_embed = self.offset_logits(offset_lstm_out)

        return page_out_embed, offset_out_embed
