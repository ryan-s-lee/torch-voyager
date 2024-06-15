import collections
import sys
import os
import struct
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as torchfunc
from torch.utils.data import DataLoader
from data import LargeTraceDataset, LargeTraceWithLabels
from loss import get_loss
from models import Prefetcher
from cProfile import Profile
from pstats import SortKey, Stats


class ModelWrapper:
    def __init__(self, config, opts) -> None:
        self.config = config
        self.page_criterion = get_loss(config["loss"])
        self.offset_criterion = get_loss(config["loss"])
        self.device = torch.device(f"cuda:{config['gpuid']}" if torch.cuda.is_available() else "cpu")
        self.num_train_batches = int(
            self.config["train-proportion"] * self.config["epoch-len"]
        )
        self.batch_size = int(self.config["batch-size"])
        self.num_val_batches = self.config["epoch-len"] - self.num_train_batches
        self.data: LargeTraceDataset | LargeTraceWithLabels  # to be set by children
        self.model: Prefetcher | nn.DataParallel
        self.page_inverter: dict
        self.optimizer = torch.optim.Optimizer
        self.scheduler = torch.optim.lr_scheduler.LRScheduler
        self.writer = None

    def train_step(
        self,
        x,  # [batch_size, seq_len, 3 (pc, page, off)]
        y,  # [batch_size, num_labels, 2 (page or offset)]
        train_stats,
        dataset_len,
    ):
        pc_in, page_in, off_in = [x[:, :, ins] for ins in (0, 1, 2)]
        page_true, offset_true = [y[:, :, po] for po in (0, 1)]
        page_true_oh, offset_true_oh = [
            torchfunc.one_hot(truth, num_classes=vocab_size)
            .sum(dim=1, dtype=torch.float)
            .clamp(max=1)
            for truth, vocab_size in (
                (page_true, self.data.page_vocab_size),
                (offset_true, 2 ** self.config["offset-bits"]), # TODO: Replace hard-coded offset sizes with a config
            )
        ]
        # compute loss
        pg_pred_oh, off_pred_oh = self.model(pc_in, page_in, off_in)
        page_loss = self.page_criterion(input=pg_pred_oh[:, -1], target=page_true_oh)
        offset_loss = self.offset_criterion(
            input=off_pred_oh[:, -1], target=offset_true_oh
        )
        step_loss = page_loss + offset_loss

        pg_pred = torch.argmax(pg_pred_oh[:, -1], dim=-1).unsqueeze_(dim=1)
        off_pred = torch.argmax(off_pred_oh[:, -1], dim=-1).unsqueeze_(dim=1)

        # record data
        batch_size = x.shape[0]
        train_stats[0] += page_loss * batch_size / dataset_len  # epoch avg page loss
        train_stats[1] += (
            offset_loss * batch_size / dataset_len
        )  # epoch avg offset loss
        train_stats[2] += (
            step_loss * batch_size / dataset_len
        )  # epoch avg combined loss

        page_correct = page_true == pg_pred
        offset_correct = offset_true == off_pred
        oll_korrect = torch.logical_and(page_correct, offset_correct)
        num_page_correct, num_offset_correct, num_combined_correct = (
            torch.sum(page_correct).item(),
            torch.sum(offset_correct).item(),
            torch.sum(oll_korrect).item(),
        )
        train_stats[3] += num_page_correct
        train_stats[4] += num_offset_correct
        train_stats[5] += num_combined_correct
        return step_loss, train_stats

    def get_scheduler(self, config, num_batches):
        if config["scheduler"] == "cosineannealing":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                # we update lr each batch, and cycle every ["lr-step"] epochs.
                T_0=config["lr-step"] * num_batches,
                T_mult=2,
                eta_min=1e-4,
            )
        elif config["scheduler"] == "multistep":
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[config["lr-step"] * i for i in range(1, 6)],
                gamma=1 / config["lr-decay"],
            )
        else:
            raise RuntimeError("Invalid scheduler")

    def print_stats(
        self,
        avg_page_loss,
        avg_offset_loss,
        avg_combined_loss,
        page_correct,
        offset_correct,
        combined_correct,
        epoch,
        mode: str,
        num_examples: int,
    ):
        # num_batches = (
        #     self.num_train_batches if mode == "train" else self.num_val_batches
        # )
        page_accuracy = page_correct / num_examples
        offset_accuracy = offset_correct / num_examples
        combined_accuracy = combined_correct / num_examples

        print(
            f"Epoch {epoch} {mode}"
            f"page loss/acc {avg_page_loss:4.3f}/{page_accuracy:4.2f}, "
            f"offset loss/acc {avg_offset_loss:4.3f}/{offset_accuracy:4.2f}, "
            f"combined addr loss/accuracy {avg_combined_loss:4.3f}/{combined_accuracy:4.2f}, "
            f"lr {self.scheduler.get_last_lr()}, ",
            file=sys.stderr,
        )
        if self.writer is not None:
            self.writer.add_scalar(mode + " page loss", avg_page_loss)
            self.writer.add_scalar(mode + " offset loss", avg_offset_loss)
            self.writer.add_scalar(mode + " combined loss", avg_combined_loss)
            self.writer.add_scalar(mode + " page accuracy", page_accuracy)
            self.writer.add_scalar(mode + " offset accuracy", offset_accuracy)
            self.writer.add_scalar(mode + " combined accuracy", combined_accuracy)


class OnlineModelWrapper(ModelWrapper):
    def __init__(self, config, opts) -> None:
        super().__init__(config, opts)
        self.trained_model = None
        self.writer = None

    def train(self) -> None:
        print("Training/inferring online", file=sys.stderr)
        sys_bs = sys.stdin.buffer
        epoch = 0
        files = collections.deque(maxlen=2)
        old_pcs = None
        old_pgs = None

        while pipe_in := sys_bs.read(32):
            # profile = Profile()
            # profile.enable()
            num_addrs, filename_len, num_uniq_pcs, num_uniq_pgs = struct.unpack(
                "4Q", pipe_in
            )
            uniq_pcs = {
                k: v
                for k, v in [
                    struct.unpack("QQ", sys_bs.read(16)) for _ in range(num_uniq_pcs)
                ]
            }
            uniq_pgs = {
                k: v
                for k, v in [
                    struct.unpack("QQ", sys_bs.read(16)) for _ in range(num_uniq_pgs)
                ]
            }
            path = sys_bs.read(filename_len)
            print(f"Found {len(uniq_pgs)} pages, {len(uniq_pcs)} pcs", file=sys.stderr)
            print("Path:", path, file=sys.stderr)
            print(f"{num_addrs} predictions to make", file=sys.stderr)
            # store this epoch's information for retrieval in the next epoch
            epoch_info = (path, num_addrs)
            files.append(epoch_info)

            # On the first epoch, we should not start training yet.
            # We need to wait for the next epoch's trace file to be complete.
            if epoch == 0:
                epoch += 1
                continue

            # Path takes on a new meaning: it is now the previous path obtained
            # from the pipe.
            path, num_addrs = files.popleft()

            sys.stdout.buffer.write(
                struct.pack(f"Q{filename_len}s", filename_len, path)
            )

            epoch_period_progress = epoch % self.config["predict-cycle-len"]
            EPOCH_SKIP = 0
            EPOCH_TRAIN_ONLY = 1
            EPOCH_TRAIN_PRED = 2
            EPOCH_PRED_ONLY = 3
            if epoch_period_progress < self.config["cycle-skip"]:
                epoch_state = EPOCH_SKIP
                print("Skipping training/predicting this epoch", file=sys.stderr)
            elif epoch_period_progress == self.config["cycle-skip"]:
                epoch_state = EPOCH_TRAIN_ONLY
                print("Training only this epoch", file=sys.stderr)
            elif self.config["cycle-skip"] == 0 or epoch_period_progress < self.config["predict-cycle-len"] - 1:
                epoch_state = EPOCH_TRAIN_PRED
                print("Training and predicting this epoch", file=sys.stderr)
            else:
                epoch_state = EPOCH_PRED_ONLY
                print("Predicting only this epoch", file=sys.stderr)

            if epoch_state != EPOCH_SKIP:
                self.data = LargeTraceDataset(
                    path, (uniq_pcs, uniq_pgs), self.config["seq-len"], self.device
                )
                self.dl = DataLoader(self.data, batch_size=self.config["batch-size"])
                self.model = Prefetcher(
                    batch_size=self.config["batch-size"],
                    seq_len=self.config["seq-len"],
                    pc_vocab_size=self.data.pc_vocab_size,
                    page_vocab_size=self.data.page_vocab_size,
                    page_out_vocab_size=self.data.page_vocab_size,
                    pc_embed_size=64,
                    page_embed_size=256,
                )
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config["lr"],
                    weight_decay=self.config["weight-decay"],
                )

                self.model.to(self.device)
                self.scheduler = self.get_scheduler(self.config, self.batch_size)
                print("created model", file=sys.stderr)

            # avg page loss, avg offset loss, full loss, page acc, offset acc, full acc
            train_stats = [0, 0, 0, 0, 0, 0]
            num_predicted_addrs = num_addrs - self.config["seq-len"]

            with tqdm(total=num_predicted_addrs) as pbar:
                # recall that if the epoch is 0, we would not have reached
                # this line. At epoch 1, train on the trace obtained during
                # epoch 0.
                if epoch_state == EPOCH_SKIP:
                    # simulate skipping predictions
                    # TODO: Factor: predict_zeros ->
                    zeros = b"\0" * 16 * (num_addrs - self.config["seq-len"])
                    print(f"writing {len(zeros)} bytes worth of 0s", file=sys.stderr)
                    sys.stdout.buffer.write(zeros)
                    # <- Factor
                elif epoch_state == EPOCH_TRAIN_ONLY:
                    # TODO: Factor: train_epoch ->
                    for x, y in iter(self.dl):
                        loss, train_stats = self.train_step(
                            x, y, train_stats, num_addrs
                        )
                        # backpropagate
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        pbar.update(len(x))
                    # <- Factor

                    pbar.refresh()
                    # simulate skipping predictions
                    # TODO: Factor: predict_zeros ->
                    zeros = b"\0" * 16 * (num_addrs - self.config["seq-len"])
                    print(f"writing {len(zeros)} bytes worth of 0s", file=sys.stderr)
                    sys.stdout.buffer.write(zeros)
                    # <- Factor
                    # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).dump_stats("profile.stats")
                    # print("wrote profile to profile.stats", file=sys.stderr)
                elif epoch_state == EPOCH_TRAIN_PRED:
                    # Make a *new* dataset over the same file, but using the old
                    # dictionaries that the trained model used
                    data2 = LargeTraceDataset(
                        path, (old_pcs, old_pgs), self.config["seq-len"], self.device
                    )
                    dl2 = DataLoader(data2, batch_size=self.config["batch-size"])
                    for (x, y), (old_x, _) in zip(iter(self.dl), iter(dl2)):
                        # <- Factor: train_epoch
                        loss, train_stats = self.train_step(
                            x, y, train_stats, num_addrs
                        )
                        # backpropagate
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        # <- Factor

                        # make predictions using the trained model
                        ins = [old_x[:, :, ins] for ins in (0, 1, 2)]
                        preds = [
                            torch.argmax(logits[:, -1], dim=-1)
                            for logits in self.trained_model(*ins)
                        ]
                        self.write_prediction(preds, sys.stdout.buffer)
                        pbar.update(len(x))
                    # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).dump_stats("profile.stats")
                    # print("wrote profile to profile.stats", file=sys.stderr)
                else:
                    # just predict; we will be predicting zeros in the next epoch.
                    data2 = LargeTraceDataset(
                        path, (old_pcs, old_pgs), self.config["seq-len"], self.device
                    )
                    dl2 = DataLoader(data2, batch_size=self.config["batch-size"])
                    for old_x, _ in iter(dl2):
                        # make predictions using the trained model
                        ins = [old_x[:, :, ins] for ins in (0, 1, 2)]
                        preds = [
                            torch.argmax(logits[:, -1], dim=-1)
                            for logits in self.trained_model(*ins)
                        ]
                        self.write_prediction(preds, sys.stdout.buffer)
                        pbar.update(len(old_x))
            # profile.disable()

            if epoch_state in [EPOCH_TRAIN_PRED, EPOCH_TRAIN_ONLY]:
                self.print_stats(*(*train_stats, epoch, "train", num_predicted_addrs))
                self.trained_model = self.model
                old_pcs = uniq_pcs
                old_pgs = uniq_pgs

                # save this model's page inverter, so that in the upcoming epoch we
                # can translate the model's predictions (which are small numbers)
                # into actual pages.
                self.page_inverter = {v: k for k, v in self.data.unique_pgs.items()}

            epoch += 1

        # if more than one epoch has passed, the last epoch did not have any predictions
        # run on its trace. Perform the predictions now.
        if epoch == 1:
            return
        path, num_addrs = files.popleft()
        sys.stdout.buffer.write(
            struct.pack(f"Q{len(path)}s", len(path), path)
        )
        if epoch_state == EPOCH_SKIP:
            # TODO: Factor: predict_zeros ->
            zeros = b"\0" * 16 * (num_addrs - self.config["seq-len"])
            print(f"writing {len(zeros)} bytes worth of 0s", file=sys.stderr)
            sys.stdout.buffer.write(zeros)
            # <- Factor
        else:
            with tqdm(total=num_addrs - self.config["seq-len"]) as pbar:
                data2 = LargeTraceDataset(
                    path, (old_pcs, old_pgs), self.config["seq-len"], self.device
                )
                dl2 = DataLoader(data2, batch_size=self.config["batch-size"])
                for old_x, _ in iter(dl2):
                    # make predictions using the trained model
                    ins = [old_x[:, :, ins] for ins in (0, 1, 2)]
                    preds = [
                        torch.argmax(logits[:, -1], dim=-1)
                        for logits in self.trained_model(*ins)
                    ]
                    self.write_prediction(preds, sys.stdout.buffer)
                    pbar.update(len(old_x))

    def write_prediction(self, predictions, buffer):
        # TODO: remove duplicate code in infer()
        page_predicts, offset_predicts = [
            pred.reshape(-1).cpu() for pred in predictions
        ]
        for page_predict, offset_predict in zip(page_predicts, offset_predicts):
            prefetch_addr = (self.page_inverter[page_predict.item()] << self.config["line-bits"] + self.config["offset-bits"]) + (
                offset_predict.item() << self.config["line-bits"]
            )
            buffer.write(struct.pack("QQ", 0, prefetch_addr))


class OfflineModelWrapper(ModelWrapper):
    def __init__(self, config, opts) -> None:
        super().__init__(config, opts)
        self.writer = SummaryWriter("runs/voyager_experiment")
        if opts.mode == "online":
            # load data later, after the tracer finishes.
            return
        self.data = self.get_data(config)
        self.dl = DataLoader(
            self.data, batch_size=config["batch-size"] if opts.mode == "train" else 1
        )
        self.model = Prefetcher(
            batch_size=config["batch-size"],
            seq_len=config["seq-len"],
            pc_vocab_size=self.data.pc_vocab_size,
            page_vocab_size=self.data.page_vocab_size,
            page_out_vocab_size=self.data.page_vocab_size,
            pc_embed_size=64,
            page_embed_size=256,
        )

        if opts.checkpoint is not None:
            ckpt_dict = torch.load(opts.checkpoint)
            print("Loaded checkpoint @ epoch", ckpt_dict["epoch"], file=sys.stderr)
            # TODO: ckpt should include other properties besides state dict (e.g. stop epoch/batch)
            self.model.load_state_dict(ckpt_dict["state"])

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight-decay"],
        )

        if torch.cuda.device_count() > 1 and config["parallel"]:
            print(f"Using {torch.cuda.device_count()} GPUs", file=sys.stderr)
            self.model = nn.DataParallel(self.model)
        else:
            print("Using only one GPU", file=sys.stderr)

        self.model.to(self.device)

        # hyperparameters

        self.scheduler = self.get_scheduler(config, self.batch_size)
        self.page_inverter = {v: k for k, v in self.data.unique_pgs.items()}

        print("created model", file=sys.stderr)

    def train(self) -> None:
        print("Training", file=sys.stderr)

        dl_iter = iter(self.dl)
        for epoch in range(self.config["num-epochs"]):
            # set model to train mode
            self.model.train()

            # avg page loss, avg offset loss, full loss, page acc, offset acc, full acc
            train_stats = [0, 0, 0, 0, 0, 0]

            for _ in tqdm(range(self.num_train_batches)):
                x, y = next(dl_iter)

                loss, train_stats = self.train_step(
                    x, y, train_stats, self.num_train_batches * self.batch_size
                )
                # backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.print_stats(*(*train_stats, epoch, "train", self.batch_size * self.num_train_batches))

            # evaluate on validation data
            self.model.eval()
            val_stats = [0, 0, 0, 0, 0, 0]  # reset stats
            for _ in tqdm(range(self.num_val_batches)):
                x, y = next(dl_iter)
                _, train_stats = self.train_step(
                    x, y, val_stats, self.num_val_batches * self.batch_size
                )
            self.print_stats(*(*train_stats, epoch, "val", self.batch_size * self.num_val_batches))

            if epoch % self.config["save-interval"] == self.config["save-interval"] - 1:
                print("Saving...", file=sys.stderr)
                save_dict = {"state": self.model.state_dict(), "epoch": epoch}
                torch.save(save_dict, self.config["outfile"])
                # breakpoint()
                self.writer.add_graph(self.model, (x[:, :, 0], x[:, :, 1], x[:, :, 2]))
            else:
                print(
                    f"Time before saving: {(epoch % self.config['save-interval']) + 1} / {self.config['save-interval']}",
                    file=sys.stderr,
                )

            self.writer.flush()
            self.scheduler.step()

    def infer(self) -> None:
        print("Infering", file=sys.stderr)
        dl_iter = iter(self.dl)

        page_inverter = {v: k for k, v in self.data.unique_pgs.items()}
        self.model.eval()
        with open(self.config["infer-out"], "wb") as infer_out, open(
            self.config["infer-out"] + ".txt", "w"
        ) as infer_out_txt:
            # TODO: print the first seq_len accesses

            i = 0
            flush_interval = self.config["flush-interval"]
            stats = os.stat(self.config["trace"])
            for _ in tqdm(range(16, stats.st_size // 16)):
                x, y = next(dl_iter)
                pc_in, page_in, off_in = [x[:, :, ins] for ins in (0, 1, 2)]
                prob_dists = [
                    every_prob_dist[0, -1]
                    for every_prob_dist in self.model(pc_in, page_in, off_in)
                ]
                page_predict, offset_predict = [
                    torch.argmax(dist, dim=-1).item() for dist in prob_dists
                ]

                prefetch_addr = (page_inverter[page_predict] << 12) + (
                    offset_predict << 6
                )
                truth = (page_inverter[y[0][0][0].item()] << 12) + (
                    y[0][0][1].item() << 6
                )
                # concatenate the predictions and shift them so it becomes the
                # first address of the prefetched block
                # write <pc><address>
                infer_out.write(struct.pack("QQ", 0, prefetch_addr))
                infer_out_txt.write(
                    f"{hex(prefetch_addr)}, {hex(truth)}, {prefetch_addr == truth}\n"
                )
                if (i + 1) % flush_interval == 0:
                    print("flushed", file=sys.stderr)
                    i = 0
                    infer_out.flush()
                i += 1

    def get_data(self, config):
        with open(config["trace-metafile"], "rb") as dicts:
            num_uniq_pcs, num_uniq_pgs = struct.unpack("QQ", dicts.read(16))
            uniq_pcs = {
                k: v
                for k, v in [
                    struct.unpack("QQ", dicts.read(16)) for _ in range(num_uniq_pcs)
                ]
            }
            uniq_pgs = {
                k: v
                for k, v in [
                    struct.unpack("QQ", dicts.read(16)) for _ in range(num_uniq_pgs)
                ]
            }
        if config["trace-format"] == "basic":
            return LargeTraceDataset(
                config["trace"],
                (uniq_pcs, uniq_pgs),
                config["seq-len"],
                self.device,
                start_example=config["warmup"]
                * config["epoch-len"]
                * config["batch-size"],
            )
        elif config["trace-format"] == "multilabel":
            return LargeTraceWithLabels(
                config["trace"],
                (uniq_pcs, uniq_pgs),
                config["seq-len"],
                self.device,
                start_example=config["warmup"]
                * config["epoch-len"]
                * config["batch-size"],
            )
        else:
            raise RuntimeError("Invalid data format")
