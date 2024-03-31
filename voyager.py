from optparse import OptionParser
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from train import train

from data import LargeTraceDataset
from models import Prefetcher

# provide nice interface to interact with the voyager:
# - train a new model
# - train a saved, pre-existing model
# - run inference using a pre-existing model
# - produce memory traces for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer(model, data: LargeTraceDataset, config, options) -> None:
    print("Infering")
    dl = DataLoader(data, batch_size=1)
    dl_iter = iter(dl)
    pc_in, page_in, offset_in, page_true, offset_true = next(dl_iter)
    seq_len = config["seq-len"]
    # invert the tables
    pc_inverter = {v: k for k, v in data.unique_pcs.items()}
    page_inverter = {v: k for k, v in data.unique_pages.items()}
    model.eval()
    with open("voyager_infer.txt", "w") as infer_out:
        # print the first seq_len accesses
        for i in range(seq_len):
            print(
                f"{hex(pc_inverter[pc_in[0][i].item()])}, {hex((page_inverter[page_in[0][i].item()] << 12) + (offset_in[0][i].item() << 6))}",
                file=infer_out,
            )
        # then, continually output accesses sequentially as before,
        # but pair them with model outputs
        # if the model output matches the intended access, print "ok" on the same line
        # otherwise, print "no" on the same line
        i = 16
        while True:
            page_predict, offset_predict = model(pc_in, page_in, offset_in)
            page_predict = torch.argmax(page_predict[0, -1], dim=-1)
            offset_predict = torch.argmax(offset_predict[0, -1], dim=-1)
            print(
                f"{hex(i):7} "
                f"{page_predict}, {offset_predict}, {page_true.item()}, {offset_true.item()}, "
                f"{hex((page_inverter[page_true.item()] << 12) + (offset_true.item() << 6))}, "
                f"{hex((page_inverter[page_predict.item()] << 12) + (offset_predict.item() << 6))}",
                "ok"
                if page_predict == page_true and offset_predict == offset_true
                else "no",
                file=infer_out,
            )

            pc_in, page_in, offset_in, page_true, offset_true = next(dl_iter)
            print(f"lines printed: {i:6}", end="\r")
            i += 1


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--mode", default="infer")
    parser.add_option("-p", "--checkpoint", default=None)
    parser.add_option("-t", "--trace", default="trace.txt")
    parser.add_option("-c", "--config", default="config.yaml")
    parser.add_option("-i", "--save-interval", type="int", default=10)
    parser.add_option("-o", "--output", default="voyager.ckpt")
    parser.add_option("-s", "--start-epoch", type="int", default=0)

    options, arguments = parser.parse_args()
    print(options)

    # TODO: read in the config file
    with open(options.config) as conf_file:
        try:
            config = yaml.safe_load(conf_file)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    print("Config:\n", config)

    # load data (commented is fixed trace data, uncommented is streamed data)
    # data = SampleTraceDataset(options.trace, config["seq-len"])
    # print("Loaded data")
    data = LargeTraceDataset(
        options.trace,
        config["seq-len"],
        start_example=options.start_epoch * config["epoch-len"] * config["batch-size"],
    )

    model = Prefetcher(
        batch_size=config["batch-size"],
        seq_len=config["seq-len"],
        pc_vocab_size=data.pc_vocab_size,
        page_vocab_size=data.page_vocab_size,
        page_out_vocab_size=data.page_vocab_size,
        pc_embed_size=64,
        page_embed_size=256,
    )

    if options.checkpoint is not None:
        ckpt_dict = torch.load(options.checkpoint)
        print("Loaded checkpoint @ epoch", ckpt_dict["epoch"])
        # TODO: ckpt should include other properties besides state dict (e.g. stop epoch/batch)
        model.load_state_dict(ckpt_dict["state"])

    print("created model")

    if torch.cuda.device_count() > 1 and config["parallel"]:
        print(f"Using {torch.cuda.device_count()} GPUs")
        mode = nn.DataParallel(model)
    else:
        print("Using only one GPU")
    model.to(device)

    if options.mode == "train":
        train(model, data, config, options)
    elif options.mode == "infer":
        infer(model, data, config, options)
    else:
        raise RuntimeError("bad mode")
