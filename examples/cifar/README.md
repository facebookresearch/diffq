# DiffQ for CIFAR-10/100

## Requirements

You must first install `diffq`, then the requirements for this example. To do so, run **from the root of the repository**:
```bash
pip install .
cd examples/cifar
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
./train.py db.name=cifar100 model=w_resnet quant.penalty=5 quant.group_size=16
```

See the Supplementary Material, Section A.4, and table B.2 for more information on the hyper-parameter used.

## LSQ

In order to run experiments with LSQ, you will first need to train a baseline model

```
./train.py db.name=cifar10 model=resnet
```

Then you can fine tune with LSQ with

```
./train.py db.name=cifar10 model=resnet dummy=ft \
	'continue_from="exp_db.name=cifar10,model=resnet"' continue_best=true \
	quant.lsq=true quant.bits=4 lr=0.01

```


## Resnet-20

To run experiments with a Resnet-20 on CIFAR-10:

```
./train.py preset=res20 quant.penalty=10 quant.group_size=16
```

## Vision Transformer

To run experiments with a Vision Transformer on CIFAR-10:

```
./train.py db.name=cifar10 model=vit quant.penalty=5 quant.group_size=16
```

## Pretrained Vision Transformer fine tune with LSQ

```
./train.py db.name=cifar10 model=vit_timm continue_best=true \
	quant.lsq=true quant.bits=4 lr=0.01
```


## License

See the file ../../LICENSE for more details.

The files `src/mobilenet.py` and `src/resnet.py` are taken from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), released as MIT.
The file `src/wide_resnet.py` is taken from [meliketoy/wide-resnet](https://github.com/meliketoy/wide-resnet.pytorch), released as MIT.
The file `src/resnet20.py` is taken from [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10), released as BSD 2-Clause "Simplified".
