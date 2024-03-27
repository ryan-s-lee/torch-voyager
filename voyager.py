from optparse import OptionParser
import torch
from torch import cuda, nn
from torch.optim._multi_tensor import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as torchfunc
import yaml
from tqdm import tqdm

from data import SampleTraceDataset
import torch.utils.data as tdata
from models import Prefetcher
from torch.utils.tensorboard.writer import SummaryWriter

# provide nice interface to interact with the voyager:
# - train a new model
# - train a saved, pre-existing model
# - run inference using a pre-existing model
# - produce memory traces for training

writer = SummaryWriter("runs/voyager_experiment")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: nn.Module,
    train_data: tdata.Dataset,
    val_data: tdata.Dataset,
    config,
    options,
) -> None:
    print("Training")

    train_dl = DataLoader(train_data, batch_size=config["batch-size"], shuffle=True)
    val_dl = DataLoader(val_data, batch_size=config["batch-size"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight-decay"],
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=config["lr-step"] * len(dl),  # we update lr each step
    #     T_mult=2,
    #     eta_min=1e-4
    # )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[config["lr-step"] * i for i in range(1, 6)],
        gamma=1 / config["lr-decay"],
    )

    def train_step(pc_in, page_in, offset_in, page_true, offset_true, train_stats, dataset_len):
        page_true_oh = torchfunc.one_hot(
            page_true, num_classes=data.page_vocab_size
        ).float()
        offset_true_oh = torchfunc.one_hot(offset_true, num_classes=64).float()
        # compute loss
        page_predict, offset_predict = model(pc_in, page_in, offset_in)
        page_loss = page_criterion(input=page_predict[:, -1], target=page_true_oh)
        offset_loss = offset_criterion(
            input=offset_predict[:, -1], target=offset_true_oh
        )
        step_loss = page_loss + offset_loss
        # record data
        train_stats[0] += page_loss * pc_in.shape[0] / dataset_len  # epoch avg page loss
        train_stats[1] += (
            offset_loss * pc_in.shape[0] / dataset_len
        )  # epoch avg offset loss
        train_stats[2] += step_loss * pc_in.shape[0] / dataset_len  # epoch avg combined loss
        page_correct = torch.argmax(page_predict[:, -1], dim=-1) == page_true
        offset_correct = torch.argmax(offset_predict[:, -1], dim=-1) == offset_true
        oll_korrect = torch.logical_and(page_correct, offset_correct)
        train_stats[3] += torch.sum(page_correct) / dataset_len  # page acc
        train_stats[4] += torch.sum(offset_correct) / dataset_len  # offset acc
        train_stats[5] += torch.sum(oll_korrect) / dataset_len  # full acc
        return step_loss, train_stats

    page_criterion = nn.CrossEntropyLoss()
    offset_criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(config["num-epochs"]):
        # set model to train mode
        model.train()

        # avg page loss, avg offset loss, full loss, page acc, offset acc, full acc
        train_stats = [0, 0, 0, 0, 0, 0]

        for pc_in, page_in, offset_in, page_true, offset_true in tqdm(iter(train_dl)):
            loss, train_stats = train_step(
                pc_in, page_in, offset_in, page_true, offset_true, train_stats, len(train_data)
            )

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        (
            avg_page_loss,
            avg_offset_loss,
            avg_combined_loss,
            page_accuracy,
            offset_accuracy,
            combined_accuracy,
        ) = train_stats
        print(
            f"Epoch {epoch}: "
            f"train page loss/acc {avg_page_loss:4.3f}/{page_accuracy:4.2f}, "
            f"train offset loss/acc {avg_offset_loss:4.3f}/{offset_accuracy:4.2f}, "
            f"train combined addr loss/accuracy {avg_combined_loss:4.3f}/{combined_accuracy:4.2f}"
            f"lr {scheduler.get_last_lr()}, "
        )
        writer.add_scalar("train page loss", avg_page_loss)
        writer.add_scalar("train offset loss", avg_offset_loss)
        writer.add_scalar("train combined loss", avg_combined_loss)
        writer.add_scalar("train page accuracy", page_accuracy)
        writer.add_scalar("train offset accuracy", offset_accuracy)
        writer.add_scalar("train combined accuracy", combined_accuracy)

        # this was useful for checking the outputs
        # print(
        #     "sample page predicts: ", torch.argmax(page_predict[:3, -1]), page_true[:3]
        # )
        # print(
        #     "sample offset predicts: ",
        #     torch.argmax(page_predict[:3, -1]),
        #     page_true[:3],
        # )

        # evaluate on validation data
        model.eval()
        train_stats = [0, 0, 0, 0, 0, 0] # reset stats

        for pc_in, page_in, offset_in, page_true, offset_true in tqdm(iter(val_dl)):
            _, train_stats = train_step(
                pc_in, page_in, offset_in, page_true, offset_true, train_stats, len(val_data)
            )

        (
            avg_page_loss,
            avg_offset_loss,
            avg_combined_loss,
            page_accuracy,
            offset_accuracy,
            combined_accuracy,
        ) = train_stats
        print(
            f"Epoch {epoch}: "
            f"val page loss/acc {avg_page_loss:4.3f}/{page_accuracy:4.2f}, "
            f"val offset loss/acc {avg_offset_loss:4.3f}/{offset_accuracy:4.2f}, "
            f"val combined addr loss/accuracy {avg_combined_loss:4.3f}/{combined_accuracy:4.2f}"
        )
        writer.add_scalar("val page loss", avg_page_loss)
        writer.add_scalar("val offset loss", avg_offset_loss)
        writer.add_scalar("val combined loss", avg_combined_loss)
        writer.add_scalar("val page accuracy", page_accuracy)
        writer.add_scalar("val offset accuracy", offset_accuracy)
        writer.add_scalar("val combined accuracy", combined_accuracy)

        if epoch % options.save_interval == 0:
            print("Saving...")
            torch.save(model.state_dict(), options.output)
            writer.add_graph(model, (pc_in, page_in, offset_in))
        else:
            print(
                f"Time before saving: {epoch % options.save_interval} / {options.save_interval}"
            )

        writer.flush()
        scheduler.step()


def infer(model, data, config, options) -> None:
    print("Infering")
    pass


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--mode", default="infer")
    parser.add_option("-p", "--checkpoint", default=None)
    parser.add_option("-t", "--trace", default="trace.txt")
    parser.add_option("-c", "--config", default="config.yaml")
    parser.add_option("-i", "--save-interval", default=10)
    parser.add_option("-o", "--output", default="voyager.ckpt")

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

    # load data
    rng = torch.Generator().manual_seed(config["rng-seed"])
    data = SampleTraceDataset(options.trace, config["seq-len"])
    train_data, val_data = tdata.random_split(
        data, [0.8, 0.2], generator=rng
    )

    if options.checkpoint is None:
        ckpt_dict = None

        # Generate model without checkpoint
        model = Prefetcher(
            batch_size=config["batch-size"],
            seq_len=config["seq-len"],
            pc_vocab_size=data.pc_vocab_size,
            page_vocab_size=data.page_vocab_size,
            page_out_vocab_size=data.page_vocab_size,
            pc_embed_size=64,
            page_embed_size=256,
        )

    else:
        print("reading checkpoint...")
        # ckpt_dict = torch.load(config.checkpoint)
        raise NotImplementedError

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        mode = nn.DataParallel(model)
    model.to(device)

    if options.mode == "train":
        train(model, train_data, val_data, config, options)
    elif options.mode == "infer":
        infer(model, data, config, options)
    else:
        raise RuntimeError("bad mode")
