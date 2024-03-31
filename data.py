import collections
import pickle
from torch.utils.data import Dataset, IterableDataset
import csv
import numpy as np
import torch
from torch.nn import functional as torchfunc
import os


class SampleTraceDataset(Dataset):
    def __init__(self, trace_file, seq_len, pc_loc=False) -> None:
        # This init is copied almost wholesale from the first portion of
        # build_and_train_network() in the original code
        super().__init__()
        self.seq_len = seq_len
        self.pc_loc = pc_loc
        unique_pcs = {-1: 0}
        unique_pages = {-1: 0}
        pc_in = []
        page_in = []
        offset_in = []
        with open(trace_file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            for row in readCSV:
                # row = row[0].split(',')
                pc, page, offset = (
                    int(row[0], 16),
                    int(row[3], 16) >> 12,
                    (int(row[3], 16) >> 6) & 0x3F,
                )
                if pc not in unique_pcs:
                    unique_pcs[pc] = len(unique_pcs)
                if page not in unique_pages:
                    unique_pages[page] = len(unique_pages)

                pc_in.append(unique_pcs[pc])
                page_in.append(unique_pages[page])
                offset_in.append(offset)

            self.pc_in = torch.tensor(np.array(pc_in)).cuda()
            self.page_in = torch.tensor(np.array(page_in)).cuda()
            self.offset_in = torch.tensor(np.array(offset_in)).cuda()

        self.page_vocab_size = len(unique_pages)
        self.pc_vocab_size = len(unique_pcs)

    def __len__(self):
        # TODO: Implement PC Localization
        if self.pc_loc:
            raise NotImplementedError
        else:
            return len(self.pc_in) - self.seq_len

    def __getitem__(self, index):
        idx_end = index + self.seq_len
        if self.pc_loc:
            raise NotImplementedError
        else:
            return (
                self.pc_in[index:idx_end],
                self.page_in[index:idx_end],
                self.offset_in[index:idx_end],
                self.page_in[idx_end],
                self.offset_in[idx_end],
            )


class TraceDataset(Dataset):
    def __init__(
        self, trace_path: str, seq_size: int, max_examples=0, offset=0
    ) -> None:
        super().__init__()

        # TODO: throw an error if max_examples < seq_size
        assert max_examples == 0 or max_examples > seq_size

        self.stats = os.stat(trace_path)
        self.seq_size = seq_size

        assert self.stats.st_size > offset
        assert self.stats.st_size - offset >= max_examples
        self.max_examples = max_examples
        self.offset = offset

        self.fd = open(trace_path, "rb")
        with open("processed.pkl", "rb") as cached_dicts:
            self.unique_pcs, self.unique_pages = pickle.load(cached_dicts)
        self.pc_vocab_size = len(self.unique_pcs)
        self.page_vocab_size = len(self.unique_pages)

    def __len__(self):
        return (
            (self.stats.st_size // 16) - self.offset - 16
            if self.max_examples == 0
            else self.max_examples
        )

    def __getitem__(self, index):
        # if training a lot, consider using mmap, moving only a few pages at a time into memory
        # this could avoid the overhead of making read calls.
        # if not shuffling examples, the pages will be accessed sequentially,
        # so consider using a try catch block to make this even faster
        self.fd.seek((index + self.offset) * 16)
        entries = self.fd.read(16 * (self.seq_size + 1))
        pc_seq, page_seq, offset_seq = [], [], []
        for i in range(0, 16 * (self.seq_size + 1), 16):
            entry = entries[i : i + 16]
            pc_seq.append(
                self.unique_pcs[int.from_bytes(entry[:8], byteorder="little")]
            )
            addr = int.from_bytes(entry[8:], byteorder="little")
            page_seq.append(self.unique_pages[addr >> 12])
            offset_seq.append((addr >> 6) & 0x3F)
        return (
            torch.tensor(pc_seq[: self.seq_size]).cuda(),
            torch.tensor(page_seq[: self.seq_size]).cuda(),
            torch.tensor(offset_seq[: self.seq_size]).cuda(),
            torch.tensor(page_seq[self.seq_size]).cuda(),
            torch.tensor(offset_seq[self.seq_size]).cuda(),
        )

    def __del__(self):
        self.fd.close()


class LargeTraceDataset(IterableDataset):
    def __init__(self, trace_path, seq_len, start_example=0) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fd = open(trace_path, "rb")
        self.fd.seek(16 * start_example, 0)
        with open("processed.pkl", "rb") as cached_dicts:
            self.unique_pcs, self.unique_pages = pickle.load(cached_dicts)
        self.pc_vocab_size = len(self.unique_pcs)
        self.page_vocab_size = len(self.unique_pages)

        # the first dimension of seq tensors is the sequence number
        # the second dimension is: 0 for pcs, 1 for page nums, and 2 for offsets.
        self.seq = torch.empty(size=(self.seq_len + 1, 3), dtype=torch.int64)
        for i in range(1, self.seq_len + 1):
            entry = self.fd.read(16)
            self.seq[i, 0] = self.unique_pcs[
                int.from_bytes(entry[:8], byteorder="little")
            ]
            addr = int.from_bytes(entry[8:], byteorder="little")
            self.seq[i, 1] = self.unique_pages[addr >> 12]
            self.seq[i, 2] = (addr >> 6) & 0x3F

        self.seq = self.seq.cuda()

    def generate(self):
        while entry := self.fd.read(16):
            addr = int.from_bytes(entry[8:], byteorder="little")
            self.seq = torch.cat(
                tensors=(
                    self.seq[1:self.seq_len+1],
                    torch.tensor(
                        [[
                            self.unique_pcs[
                                int.from_bytes(entry[:8], byteorder="little")
                            ],
                            self.unique_pages[addr >> 12],
                            (addr >> 6) & 0x3F,
                        ]]
                    ).cuda(),
                ),
                dim=0,
            )
            yield (
                self.seq[: self.seq_len, 0],
                self.seq[: self.seq_len, 1],
                self.seq[: self.seq_len, 2],
                self.seq[self.seq_len, 1],
                self.seq[self.seq_len, 2],
            )

    def __iter__(self):
        return iter(self.generate())
