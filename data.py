from torch.utils.data import Dataset
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
                self.offset_in[idx_end]
            )


class TraceDataset(Dataset):
    def __init__(self, trace_path: str, batch_size: int) -> None:
        super().__init__()

        self.stats = os.stat(trace_path)
        self.batch_size = batch_size
        self.data_arr = np.memmap(trace_path, mode="r", dtype=np.uint8, shape=(self.stats.st_size // 2, 2))

        # TODO: Remove below line, as it is a test
        print(self.data_arr[0])

        self.unique_pcs = np.unique(self.data_arr[:, 0])
        self.unique_pages = np.unique(self.data_arr[:, 1])
        self.page_vocab_size = 0
        self.page_vocab_size = 0

    def __len__(self):
        return self.stats.st_size / 16 / self.batch_size

    def __getitem__(self, index):
        return super().__getitem__(index)
