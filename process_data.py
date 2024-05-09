from optparse import OptionParser
import os
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file")
    parser.add_option("-m", "--multilabel")

    opts, args = parser.parse_args()

    unique_pc = {-1: 0}
    unique_page = {-1: 0}
    stats = os.stat(opts.file)
    with open(opts.file, "rb") as train_data:
        for _ in tqdm(range(stats.st_size // 16)):
            # for each entry (that is, access in the trace):
            # obtain its program counter, and page
            entry = train_data.read(16)
            addr = int.from_bytes(entry[:8], byteorder="little")
            pc = int.from_bytes(entry[8:], byteorder="little")
            page = addr >> 12
            # offset = (addr >> 6) & 0x3F

            # maintain that the every program counter and page seen exists
            # as a key in a dictionary and that its value is the order in
            # which it was seen.
            if pc not in unique_pc:
                unique_pc[pc] = len(unique_pc)

            if page not in unique_page:
                unique_page[page] = len(unique_page)

    print("num unique pcs:", len(unique_pc), "| num unique pages: ", len(unique_page))
    with open("processed.pkl", "wb") as outfile:
        pickle.dump((unique_pc, unique_page), outfile)
