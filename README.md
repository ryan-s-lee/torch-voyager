# torch-voyager
Pytorch implementation of Voyager, for modern times.

## Online Training
```
python voyager.py -m online < [/path/to/pipe/from/trace] > [/path/to/output/pipe]

```

Voyager will read from the `pipe/from/trace` (whatever you set it to).
The tracer will output to this pipe, and the output will consist of
the path to the actual trace file, how many unique pcs/pgs there are,
and a dictionary that enumerates each unique pc found as well as 
another dictionary with a separate enumeration for each page.

Voyager cycles between four states. The first, EPOCH_SKIP, indicates
that Voyager is not training/predicting at all. This state is used to
limit how much of the trace we are actually predicting on, which 
is necessary to gather data in a timely manner since neural prefetching 
is so slow. In the EPOCH_TRAIN_ONLY phase, Voyager will only train a 
single model. This phase should last for one cycle. In the EPOCH_TRAIN_PRED phase, Voyager will use the most
recently trained model and use it to make predictions on the current
epoch, and it trains another fresh model. At the end of an epoch in 
this phase, the model last used to make predictions is thrown out, and
the newly trained model is used to predict in the next epoch. In the
EPOCH_PRED_ONLY phase, the model doesn't train a model, and only uses
the last-trained model to make predictions. This phase should only
last one cycle. How many epochs it takes to cycle through all four states is
determined by `predict-cycle-len` in config.yaml. How many epochs the skip
phase lasts is determined by `cycle-skip`. The larger `cycle-skip` is relative
to `predict-cycle-len`, the faster you will go through the simulation, but
you will make predictions on less of the trace (and thus gather less information).


To set how many bits are in the block offset, assign `line-bits` in 
`config.yaml`. To set how many bits are in the set index, assign 
`offset-bits` in `config.yaml` Note that these values have to be the 
same as `LINE_BITS` and `OFFSET_BITS` assigned in the tracer's 
`heap_tracer.h`. Play around with these to see how they affect performance,
train/eval speed, and perhaps memory consumption.

## Train (likely does not currently work)
```
python voyager.py -m train [TODO]

```

During training, the model treats the trace file as a stream
of examples. It will treat every `batch-size` examples as a batch, 
whatever you set it to in `config.yaml`. It will 
treat `epoch-len` separate, consecutive batches as an epoch.
When an epoch is finished, the training loop will NOT rerun training
on the data in the epoch, but will instead draw new examples from the 
trace file for use in the next epoch.

The output (by default voyager.ckpt) will be a dictionary containing
the model and the epoch the model was saved at. By default, 
the output will be written to every 10 epochs.

### Trace File Format
Trace files are sequences of examples, where examples represent
memory accesses. Each example is 16 bytes long. The first 8 bytes
represents the program counter. The second 8 bytes represents the
accessed address. During training, we use the PC and extract the
page and offset of the would-be-cached block within the page for
use as features.

Other options include:
- `-s, --start-epoch`: Determines which example in the trace file to
start reading from. The example will always be the start of an epoch
(hence, start-epoch). The nth example will be chosen where 
`n = start-epoch * epoch-len * batch-size`. Default: 0
- `-p, --checkpoint`: If set, assigns sets the model weights to the
ones stored in the checkpoint before training. Epochs currently 
start at 0 regardless of whether there is a checkpoint, but this
will be changed at a later date.
- `-o, --output`: The file name to output checkpoints to.
- `-s, --save-interval`: How many epochs to train before saving. Default: `10`
- `-c, --config`: The path to the config file to use. Default: `config.yaml`

## Infer
```
python voyager.py -m infer -t \<trace\> -p \<saved-model\>
```

During inference, a text file called `voyager_infer.txt` will be 
written to containing the results of inference. The first 16 lines
will be the 16 accesses used to make the first prediction. The
format for these lines is: 
```
\<pc\>, \<accessed-address\>
```

The rest of the lines contain the actual predictions. The format
is as follows (note that there is no program counter here; this
may be added at a later date):
```
hex-counter true-address predicted-address correctness
```

the `correctness` value is `ok` if the true and predicted addresses
match, and `no` if they are different.

Other options:
- `-s, --start-epoch`: Determines which example in the trace file to
start reading from. The example will always be the start of an epoch
(hence, start-epoch). The nth example will be chosen where 
`n = start-epoch * epoch-len * batch-size`. Default: 0. Note: this is
especially useful in inference as it can be used to run inference on
examples at timesteps past what the model used to train.
- `-c, --config`: The path to the config file to use. Default: `config.yaml`
