from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as torchfunc
from torch.utils.data import DataLoader
from data import LargeTraceDataset

writer = SummaryWriter("runs/voyager_experiment")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: nn.Module,
    train_data: LargeTraceDataset,
    config,
    options,
) -> None:
    print("Training")

    dl = DataLoader(train_data, batch_size=config["batch-size"])

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

    def train_step(
        pc_in, page_in, offset_in, page_true, offset_true, train_stats, dataset_len
    ):
        # print(pc_in)
        page_true_oh = torchfunc.one_hot(
            page_true, num_classes=train_data.page_vocab_size
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
        train_stats[0] += (
            page_loss * pc_in.shape[0] / dataset_len
        )  # epoch avg page loss
        train_stats[1] += (
            offset_loss * pc_in.shape[0] / dataset_len
        )  # epoch avg offset loss
        train_stats[2] += (
            step_loss * pc_in.shape[0] / dataset_len
        )  # epoch avg combined loss
        page_correct = torch.argmax(page_predict[:, -1], dim=-1) == page_true
        offset_correct = torch.argmax(offset_predict[:, -1], dim=-1) == offset_true
        oll_korrect = torch.logical_and(page_correct, offset_correct)
        num_page_correct, num_offset_correct, num_combined_correct = (
            torch.sum(page_correct).item(),
            torch.sum(offset_correct).item(),
            torch.sum(oll_korrect).item(),
        )
        # print(num_page_correct, num_offset_correct, num_combined_correct)
        train_stats[3] += num_page_correct  #  / dataset_len  # page acc
        train_stats[4] += num_offset_correct  #  / dataset_len  # offset acc
        train_stats[5] += num_combined_correct  #  / dataset_len  # full acc
        return step_loss, train_stats

    page_criterion = nn.CrossEntropyLoss()
    offset_criterion = nn.CrossEntropyLoss()

    # train
    dl_iter = iter(dl)
    for epoch in range(config["num-epochs"]):
        # set model to train mode
        model.train()

        # avg page loss, avg offset loss, full loss, page acc, offset acc, full acc
        train_stats = [0, 0, 0, 0, 0, 0]

        num_train_batches = int(config["train-proportion"] * config["epoch-len"])
        batch_size = int(config["batch-size"])
        for _ in tqdm(range(num_train_batches)):
            pc_in, page_in, offset_in, page_true, offset_true = next(dl_iter)
            loss, train_stats = train_step(
                pc_in,
                page_in,
                offset_in,
                page_true,
                offset_true,
                train_stats,
                num_train_batches * batch_size,
            )

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # breakpoint()

        (
            avg_page_loss,
            avg_offset_loss,
            avg_combined_loss,
            page_correct,
            offset_correct,
            combined_correct,
        ) = train_stats
        page_accuracy = page_correct / (batch_size * num_train_batches)
        offset_accuracy = offset_correct / (batch_size * num_train_batches)
        combined_accuracy = combined_correct / (batch_size * num_train_batches)

        print(
            f"Epoch {epoch} train: "
            f"page loss/acc {avg_page_loss:4.3f}/{page_accuracy:4.2f}, "
            f"offset loss/acc {avg_offset_loss:4.3f}/{offset_accuracy:4.2f}, "
            f"combined addr loss/accuracy {avg_combined_loss:4.3f}/{combined_accuracy:4.2f}, "
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
        train_stats = [0, 0, 0, 0, 0, 0]  # reset stats
        num_val_batches = config["epoch-len"] - num_train_batches
        for _ in tqdm(range(num_val_batches)):
            pc_in, page_in, offset_in, page_true, offset_true = next(dl_iter)
            _, train_stats = train_step(
                pc_in,
                page_in,
                offset_in,
                page_true,
                offset_true,
                train_stats,
                num_val_batches * batch_size,
            )

        (
            avg_page_loss,
            avg_offset_loss,
            avg_combined_loss,
            page_correct,
            offset_correct,
            combined_correct,
        ) = train_stats
        page_accuracy = page_correct / (batch_size * num_val_batches)
        offset_accuracy = offset_correct / (batch_size * num_val_batches)
        combined_accuracy = combined_correct / (batch_size * num_val_batches)
        print(
            f"Epoch {epoch} val: "
            f"page loss/acc {avg_page_loss:4.3f}/{page_accuracy:4.2f}, "
            f"offset loss/acc {avg_offset_loss:4.3f}/{offset_accuracy:4.2f}, "
            f"combined addr loss/accuracy {avg_combined_loss:4.3f}/{combined_accuracy:4.2f}"
        )
        writer.add_scalar("val page loss", avg_page_loss)
        writer.add_scalar("val offset loss", avg_offset_loss)
        writer.add_scalar("val combined loss", avg_combined_loss)
        writer.add_scalar("val page accuracy", page_accuracy)
        writer.add_scalar("val offset accuracy", offset_accuracy)
        writer.add_scalar("val combined accuracy", combined_accuracy)

        if epoch % options.save_interval == options.save_interval - 1:
            print("Saving...")
            save_dict = {"state": model.state_dict(), "epoch": epoch}
            torch.save(save_dict, options.output)
            writer.add_graph(model, (pc_in, page_in, offset_in))
        else:
            print(
                f"Time before saving: {(epoch % options.save_interval) + 1} / {options.save_interval}"
            )

        writer.flush()
        scheduler.step()
