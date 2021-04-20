# DiffQ for DeiT: Data-efficient Image Transformers

This code has been adapted from https://github.com/facebookresearch/deit.

## Requirements

You must first install `diffq`, and apply the patch to the mainstream DeiT branch. To do so, run **from the root of the repository**:
```
pip install .
make examples
cd examples/deit
pip install -r requirements.txt
```

## Training with DiffQ:

To train, run
```
./distributed_train.sh {NUMBER_OF_GPUS} --data-path {PATH_TO_IMNET} [ARGS]
```

The folder `{PATH_TO_IMNET}` should contain `test`, `train` and `val` subfolders.
Note that the batch size is provided per GPU, and that we used 8 GPUs for training.

To retrain the baseline, pass no arguments. To train a QAT model, pass

```
python main.py --data-path {PATH_TO_IMNET} --qat --bits {NUMBER_OF_BITS}
```

To train a DiffQ model, use
```
python main.py --data-path {PATH_TO_IMNET} --penalty={penalty level} --group_size={group size}
```