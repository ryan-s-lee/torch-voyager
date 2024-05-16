from optparse import OptionParser
import torch
import yaml
from train import OfflineModelWrapper, OnlineModelWrapper
import sys


# provide nice interface to interact with the voyager:
# - train a new model
# - train a saved, pre-existing model
# - run inference using a pre-existing model
# - produce memory traces for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--mode", default="infer")
    parser.add_option("-p", "--checkpoint", default=None)
    parser.add_option("-c", "--config", default="config.yaml")

    options, arguments = parser.parse_args()
    print(options, file=sys.stderr)

    # TODO: read in the config file
    with open(options.config) as conf_file:
        try:
            config = yaml.safe_load(conf_file)
        except yaml.YAMLError as exc:
            print(exc, file=sys.stderr)
            exit()
    print("Config:\n", config, file=sys.stderr)


    if options.mode == "train":
        model_wrapper = OfflineModelWrapper(config, options)
        model_wrapper.train()
    elif options.mode == "infer":
        model_wrapper = OfflineModelWrapper(config, options)
        model_wrapper.infer()
    elif options.mode == "online":
        model_wrapper = OnlineModelWrapper(config, options)
        model_wrapper.train()
    elif options.mode == "compile":
        sm = torch.jit.script(model_wrapper.model)
        sm.save(config["outfile"])
    else:
        raise RuntimeError("bad mode")
