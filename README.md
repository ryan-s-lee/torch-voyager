# torch-voyager
Pytorch implementation of Voyager, for modern times.

## Train
```
python voyager.py -m train -t \<trace\>

```

During training, the model treats the trace file as a stream
of examples. It will treat every 256 examples as a batch 
(or whatever you set batch-size to in the config). It will 
treat 200 (epoch-len) separate, consecutive batches as an epoch.
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
