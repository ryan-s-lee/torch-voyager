import lzma
import sys
import os
from array import array
from optparse import OptionParser
import collections
from tqdm import tqdm
import itertools

parser = OptionParser()
parser.add_option("-s", "--max-search-depth", default=-1)

if len(sys.argv) != 3:
    print("Usage: get_multilabel_trace.py <trace-path> <output-path>")

trace_filepath = sys.argv[1]
out_path = sys.argv[2]
if trace_filepath.endswith(".xz"):
    t = lzma.open(trace_filepath, "wb")
else:
    t = open(trace_filepath, "rb")

if out_path.endswith(".xz"):
    o = lzma.open(out_path, "wb")
else:
    o = open(out_path, "wb")


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
    arr = array("Q", t.read(16))
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
        pc_to_line[new_line[PC_ID]][INC_LINE][PC_ADDR_ID] = new_line[CUR_ADDR_ID]
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

stats = os.stat(trace_filepath)
pbar = tqdm(total=stats.st_size // 16)
while entry := t.read(16):
    pbar.update()
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
    line_to_set_co[INC_LINE][CO_ADDR_ID] = max(last_ten_freq, key=last_ten_freq.get)
    line_to_set_co[INC_CTR] -= 1

    # the first number is how many addresses need to be added before
    # we can write the line. ignore BB address.
    incomplete_lines.append(new_line_with_addr_count)

    # keep popping/writing head if it has added all necessary addresses,
    # or until queue length is <= max search distance.
    while len(incomplete_lines) > max_search_dist or incomplete_lines[0][INC_CTR] == 0:
        line_to_write = incomplete_lines.popleft()[INC_LINE]
        array("Q", line_to_write).tofile(o)
