import sys
import collections
import pickle
from torch.utils.data import IterableDataset
import torch
from array import array
import struct


class LargeTraceDataset(IterableDataset):
    def __init__(self, trace_path, dicts, seq_len, device, start_example=0) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fd = open(trace_path, "rb")
        self.fd.seek(16 * start_example, 0)

        self.unique_pcs, self.unique_pgs = dicts
        self.pc_vocab_size = len(self.unique_pcs)
        self.page_vocab_size = len(self.unique_pgs)
        self.device = device

        # the first dimension of seq tensors is the sequence number
        # the second dimension is: 0 for pcs, 1 for page nums, and 2 for offsets.
        self.seq = torch.empty(size=(self.seq_len + 1, 3), dtype=torch.int64)
        for i in range(1, self.seq_len + 1):
            entry = self.fd.read(16)
            pc, addr = struct.unpack("QQ", entry)
            # TODO: see if doing this in-place is faster
            self.seq[i] = torch.tensor(
                [self.unique_pcs[pc], self.unique_pgs[addr >> 12], (addr >> 6) & 0x3F]
            )

        self.seq = self.seq.to(self.device)

    def generate(self):
        while entry := self.fd.read(16):
            pc, addr = struct.unpack("QQ", entry)
            pc = self.unique_pcs[pc]
            page = self.unique_pgs[addr >> 12]
            offset = (addr >> 6) & 0x3F
            # seq stores the 17 most recent accesses.
            # They are represented in the numpy array as:
            # 0: pcs
            # 1: page
            # 2: offset
            # we the programmer to split the address into page/address later.
            self.seq = torch.vstack(
                tensors=(
                    self.seq[1 : self.seq_len + 1],
                    torch.tensor([[pc, page, offset]]).to(self.device),
                ),
            )

            # The first 16 entries in seq are the input sequence
            # The final entry is the one to predict.
            # TODO: convert prediction entry to onehot.
            yield (
                self.seq[: self.seq_len],
                self.seq[self.seq_len, 1:].reshape(1, 2),
            )

    def __iter__(self):
        return iter(self.generate())


PC_ID = 0
CUR_ADDR_ID = 1
GLB_ADDR_ID = 2
PC_ADDR_ID = 3
SPATIAL_ADDR_ID = 4
BB_ADDR_ID = 5
CO_ADDR_ID = 6

INC_CTR = 0
INC_LINE = 1

NUM_TARGET_ADDRS = 5


class LargeTraceWithLabels(IterableDataset):
    def __init__(
        self, trace_path, trace_metafile, seq_len, device, start_example=0
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        if trace_path == "stdin":
            self.fd = sys.stdin.buffer
        else:
            self.fd = open(trace_path, "rb")
            self.fd.seek(16 * start_example, 0)
        with open(trace_metafile, "rb") as cached_dicts:
            self.unique_pcs, self.unique_pgs = pickle.load(cached_dicts)
        self.pc_vocab_size = len(self.unique_pcs)
        self.page_vocab_size = len(self.unique_pgs)
        self.device = device

    def generate(self):
        max_search_dist = 16384

        # create a queue to store unfinished lines
        incomplete_lines = collections.deque()

        # map seen pcs to their position in the incomplete set
        # used for obtaining next address of pc
        pc_to_line = {}
        last_ten = collections.deque()
        last_ten_freq = {}

        # prime the stream
        for i in range(10):
            arr = array("Q", self.fd.read(16))
            new_line = arr.tolist() + [0 for _ in range(NUM_TARGET_ADDRS)]

            if i != 0:
                incomplete_lines[-1][INC_LINE][GLB_ADDR_ID] = new_line[CUR_ADDR_ID]
                incomplete_lines[-1][INC_CTR] -= 1

            # search for a spatial address
            for line in reversed(incomplete_lines):
                if (
                    line[INC_LINE][SPATIAL_ADDR_ID] == 0
                    and line[INC_LINE][CUR_ADDR_ID] - new_line[CUR_ADDR_ID] < 256
                ):
                    line[INC_LINE][SPATIAL_ADDR_ID] = new_line[CUR_ADDR_ID]
                    line[INC_CTR] -= 1
                    break

            # If there is a preceding line with the same PC,
            # update its next PC address entry
            # regardless, set up the current PC to obtain the next PC address.
            if new_line[PC_ID] in pc_to_line:
                pc_to_line[new_line[PC_ID]][INC_LINE][PC_ADDR_ID] = new_line[
                    CUR_ADDR_ID
                ]
                pc_to_line[new_line[PC_ID]][INC_CTR] -= 1

            new_line_with_addr_count = [NUM_TARGET_ADDRS - 1, new_line]
            pc_to_line[new_line[PC_ID]] = new_line_with_addr_count

            # the first number is how many addresses need
            # to be added before we can write the line.
            # ignore BB addresses.
            incomplete_lines.append(new_line_with_addr_count)
            last_ten.append(new_line_with_addr_count)
            last_ten_freq[new_line[CUR_ADDR_ID]] = (
                last_ten_freq.get(new_line[CUR_ADDR_ID], 0) + 1
            )
            # breakpoint()

        while entry := self.fd.read(16):
            arr = array("Q", entry)
            # obtain pc and memory access address (trivial)
            new_line = arr.tolist() + [0 for _ in range(NUM_TARGET_ADDRS)]

            # update previous line's next global access
            incomplete_lines[-1][INC_LINE][GLB_ADDR_ID] = new_line[CUR_ADDR_ID]
            incomplete_lines[-1][INC_CTR] -= 1

            # search for a spatial address
            # note that this is the main bottleneck in performance,
            # as it scales linearly with number of traces entries (up to
            # the search distance)
            for line in reversed(incomplete_lines):
                if (
                    line[INC_LINE][SPATIAL_ADDR_ID] == 0
                    and line[INC_LINE][CUR_ADDR_ID] - new_line[CUR_ADDR_ID] < 256
                ):
                    line[INC_LINE][SPATIAL_ADDR_ID] = new_line[CUR_ADDR_ID]
                    line[INC_CTR] -= 1
                    break

            # If there is a preceding line with the same PC,
            # update its next PC address entry
            # regardless, set up the current PC to obtain the next PC address.
            if new_line[PC_ID] in pc_to_line:
                line_to_update_pc = pc_to_line[new_line[PC_ID]]
                line_to_update_pc[INC_LINE][PC_ADDR_ID] = new_line[CUR_ADDR_ID]
                line_to_update_pc[INC_CTR] -= 1

            new_line_with_addr_count = [NUM_TARGET_ADDRS - 1, new_line]
            pc_to_line[new_line[PC_ID]] = new_line_with_addr_count

            # Skip Basic Block address

            # Compute most common of the 9 previous elements and this one,
            # then update the 10th-previous line's co-occurence address.
            line_to_set_co = last_ten.popleft()
            last_ten.append(new_line_with_addr_count)
            last_ten_freq[line_to_set_co[INC_LINE][CUR_ADDR_ID]] -= 1
            if last_ten_freq[line_to_set_co[INC_LINE][CUR_ADDR_ID]] == 0:
                del last_ten_freq[line_to_set_co[INC_LINE][CUR_ADDR_ID]]
            last_ten_freq[new_line[CUR_ADDR_ID]] = (
                last_ten_freq.get(new_line[CUR_ADDR_ID], 0) + 1
            )
            line_to_set_co[INC_LINE][CO_ADDR_ID] = max(
                last_ten_freq, key=last_ten_freq.get
            )
            line_to_set_co[INC_CTR] -= 1

            # the first number is how many addresses need to be added before
            # we can write the line. ignore BB address.
            incomplete_lines.append(new_line_with_addr_count)

            # keep popping/writing head if it has added all necessary addresses,
            # or until queue length is <= max search distance.
            while (
                len(incomplete_lines) > max_search_dist
                or incomplete_lines[0][INC_CTR] == 0
            ):
                yield incomplete_lines.popleft()[INC_LINE]

    def addr_line_to_pageoff(self, line):
        return [self.unique_pcs[line[PC_ID]]] + [
            f(x)
            for x in line[PC_ID + 1 :]
            for f in (lambda x: self.unique_pgs[x >> 12], lambda x: (x >> 6) & 0x3F)
        ]

    def generator2(self):
        seq = collections.deque()
        file_iter = iter(self.generate())
        for _ in range(self.seq_len):
            addr_line = next(file_iter)
            page_offs = self.addr_line_to_pageoff(addr_line)
            breakpoint()
            seq.append(page_offs)

        for line in file_iter:
            seq.append(self.addr_line_to_pageoff(line))
            focus_block = torch.tensor(seq).to(self.device)
            x = focus_block[: self.seq_len, :3]
            y = focus_block[self.seq_len, 3:].reshape(5, 2)
            yield x, y
            seq.popleft()

    def __iter__(self):
        # TODO: Make an iterator, over the iterator! Have it initially draw 16
        # entries as the first input sequence.
        return iter(self.generator2())
