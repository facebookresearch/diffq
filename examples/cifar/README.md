# DiffQ for CIFAR-10/100

## Requirements

First install the requirements with

```
pip install -r requirements.txt
```

## Training

In order to train a model you can run

```
./train.py db.name={DATASET} model={MODEL}
```
with DATASET either `cifar10` or `cifar100` and model one of
`resnet` (ResNet 18), `mobilenet` (MobileNet), or `w_resnet` (Wide ResNet).
The datasets will be automatically downloaded in the `./data` folder, and
the checkpoints stored in the `./outputs` folder.


## QAT

To train with qat,
```
./train.py db.name={DATASET} model={MODEL} quant.bits={BITS} quant.qat=True
```

for instance

```
./train.py db.name=cifar100 model=mobilenet quant.bits=3 quant.qat=True
```

## DiffQ

To train with diffq, with a given model size penalty and group size.
```
./train.py db.name={DATASET} model={MODEL} quant.penalty={PENALTY} quant.group_size={GROUP_SIZE}
```

for instance

```
./train.py db.name=cifar100 model=w_resnet quant.penalty=0.01 quant.group_size=16
```


## License

See the file ../../LICENSE for more details.

The files `src/mobilenet.py` and `src/resnet.py` are taken from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), released as MIT.
The file `src/wide_resnet.py` is taken from [meliketoy/wide-resnet](https://github.com/meliketoy/wide-resnet.pytorch), released as MIT.