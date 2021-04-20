# DiffQ for DeiT: Data-efficient Image Transformers

This code has been adapted from https://github.com/facebookresearch/fairseq.

## Requirements

You must first install `diffq`, and apply the patch to the mainstream DeiT branch. To do so, run **from the root of the repository**:
```bash
pip install .
make examples
cd examples/fairseq
pip install .
```

## Training with DiffQ:

To train a transformer with Fairseq and Diffq, adapt and run the following:
```bash
export WIKITEXT_PATH=/path/to/wikitext-103  # where to store models
export SAVE_PATH=/tmp/fairseq_diffq  # where to store checkpoints
./run.py [-d] [ARGS]
```

The `-d` flag will use distributed training locally. On a Slurm cluster with submitit, you can also pass the `-G NB_GPUS` option to schedule a job.
As Fairseq automatically scale the batch size depending on the number of gpus, and we used 24 gpus for training, you will also need to pass the `--update-freq=RATIO` option, to scale the number of optimizer step to match our setting. For instance, if you are using 8 gpus, you should pass `--update-freq=3` to exactly reproduce our results.

In order to train QAT 4 and QAT 8 models:
```bash
./run.py -b 4
./run.py -b 8
```
For DiffQ model with penalty level 1 or 10, and group size 16:
```bash
./run.py -p 1 -g 16
./run.py -p 10 -g 16
```

### Experiment names and folders.

Note that interrupted experiments will automatically resume checkpoints.
Logs will be stored inside `SAVE_PATH/experiment_name/` with the experiment name depending on the arguments passed.
For instance, the QAT 4 model will be named `exp_bits=4`.

The logs will also be in that folder, following the format `trainer.{RUN_IDX}.log.{WORKER_IDX}`. To get only validation errors for each epoch, along with model size, run from within
the experiment folder:
```bash
cat train.*.log.0| grep "'valid' subset"
```

## Evaluating models

In order to evaluate models on the test set, you can use the `examples/fairseq/eval.sh` script:

```bash
./eval.sh EXPERIMENT_NAME  # evaluate on test set, no activation quantization.
./eval.sh EXPERIMENT_NAME minmax_pc # with min max per channel activation quantization.
./eval.sh EXPERIMENT_NAME histogram # with histogram activation quantization.
```

