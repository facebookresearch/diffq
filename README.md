# Differentiable Model Compression via Pseudo Quantization Noise
![linter badge](https://github.com/facebookresearch/diffq/workflows/linter/badge.svg)
![tests badge](https://github.com/facebookresearch/diffq/workflows/tests/badge.svg)
![cov badge](https://github.com/facebookresearch/diffq/workflows/cov%3E90%25/badge.svg)

DiffQ performs differentiable quantization using pseudo quantization noise.
It can automatically tune the number of bits used per weight or group of weights,
in order to achieve a given trade-off between model size and accuracy.

Go read [our paper][paper] for more details.


## What's up?

See [the changelog](CHANGELOG.md) for details on releases.

- 2022-08-24: v0.2.3: fixed a bug when loading old quantized states.
- 2021-11-25: version 0.2.2: adding support for torchscript.

## Requirements

DiffQ requires Python 3.7, and a reasonably recent version of PyTorch (1.7.1 ideally).
To install DiffQ, you can run from the root of the repository:

```
pip install .
```

You can also install directly from PyPI with `pip install diffq`.


## Usage

```python
import torch
from torch.nn import functional as F
import diffq
from diffq import DiffQuantizer

model = MyModel()
optim = ...  # The optimizer must be created before the quantizer
quantizer = DiffQuantizer(model)
quantizer.setup_optimizer(optim)

# Distributed data parallel must be created after DiffQuantizer!
dmodel = torch.distributed.DistributedDataParallel(...)

penalty = 1e-3
model.train()  # call model.eval() on eval to automatically use true quantized weights.
for batch in loader:
    ...
    optim.zero_grad()

    # The `penalty` parameter here will control the tradeoff between model size and model accuracy.
    loss = F.mse_loss(x, y) + penalty * quantizer.model_size()
    optim.step()

# To get the true model size with when doing proper bit packing.
print(f"Model is {quantizer.true_model_size():.1f} MB")

# When you want to dump your final model:
torch.save(quantizer.get_quantized_state(), "some_file.th")

# You can later load back the model with
model = MyModel()
diffq.restore_quantized_state(model, torch.load("some_file.th"))

# For DiffQ models, we support exporting the model to Torscript with optimal storage.
# Once loaded, the model will be stored in fp32 in memory (int8 support coming up).
from diffq.ts_export import export
export(quantizer, 'quantized.ts')
```

## Documentation

See the [API documentation][api] for detailed documentation.
We cover hereafter a few aspects.

### Quantizer object

A Quantizer is attached to a model at its creation.
All Quantizer objects provide the same basic capabilities:
- automatically switches to quantized weights on the forward if the model is in eval mode.
- quantizer-specific code on training forward (e.g. STE for UniformQuantizer with QAT,
 noise injection for DiffQ).
- provide access to the quantized model size and state.

### Quantized size and state

The method `quantizer.model_size()` provide a differentiable model size (for DiffQ),
  while `quantizer.true_model_size()` provide the true, optimally bit-packed, model size
  (non differentiable).
  With `quantizer.compressed_model_size()` you can get the model size using `gzip`.
  This can actually be larger than the true model size, and reveals interesting
  information on the entropy usage of a specific quantization method.

The bit-packed quantized state is obtained with `quantizer.get_quantized_state()` ,
and restored with `quantizer.restore_quantized_state()`.
Bit packing is optimized for speed and can suffer from some overhead
(in practice no more than 120B for Uniform and LSQ, and not more than 1kB for DiffQ).

If you do not have access to the original quantizer, for instance at inference time,
you can load the state with `diffq.restore_quantized_state(model, quantized_state)`.

### Quantizer and optimization

Some quantizer will add extra optimizable parameters (DiffQuantizer and LSQ).
Those parameters can require different optimizers or hyper-parameters than
the main model weights.
Typically, DiffQ bits parameters are always optimized with Adam.
For that reason, you should always create the main optimizer **before**
the quantizer. You can then setup the quantizer with this optimizer or another:

```python
model = MyModel(...)
opt = torch.optim.Adam(model.parameters())
quantizer = diffq.DiffQuantizer(model)
quantizer.setup_optimizer(opt, **optim_overrides)
```

This offers the freedom to use a separate hyper-params. For instance, `DiffQuantizer`
will always deactivate weight_decay for the bits parameters.

If the main optimizer is SGD, it is advised to have a second Adam optimizer
for the quantizer.

**Warning**: you must always wrap your model with `DistributedDataParallel`
after having created the quantizer, otherwise the quantizer parameters won't be optimized!

### TorchScript support

At the moment the TorchScript support is experimental. We support saving
the model with TorchScript to disk with optimal storage. Once loaded, the model
is stored in FP32 in memory. We are working towards adding support for int8
in memory. See the `diffq.ts_export.export` function in the API.

## Examples

We provide three examples in the `examples/` folder. One is for CIFAR-10/100,
using standard architecture such as Wide-ResNet, ResNet or MobileNet.
The second is based on the [DeiT][deit] visual transformer.
The third is a language modeling task on Wikitext-103, using [Fairseq][fairseq]

The DeiT and Fairseq examples are provided as a patch on the original codebase at a specific
commit. You can initialize the git submodule and apply the patches by running

```
make examples
```

For more details on each example, go checkout their specific READMEs:

- [CIFAR README](examples/cifar/README.md)
- [DeiT README](examples/DEIT_README.md)
- [Fairseq README](examples/FAIRSEQ_README.md)


## Installation for development

This will install the dependencies and a `diffq` in developer mode (changes to the files
will directly reflect), along with the dependencies to run unit tests.
```
pip install -e '.[dev]'
```

### Updating the patch based examples

In order to update the patches, first run `make examples` to properly initialize the sub repos. Then perform all the changes you want, commit them and run `make patches`. This will update the patches for each repo. Once this is done, and you checked that all the changes you did are properly included in the new patch files, you can run `make reset` (this will remove all your changes you did from the submodules, so do check the patch files before calling this) before calling `git add -u .; git commit -m "my changes"` and pushing.


### Test

You can run the unit tests with
```
make tests
```

## Citation

If you use this code or results in your paper, please cite our work as:

```
@article{defossez2021differentiable,
  title={Differentiable Model Compression via Pseudo Quantization Noise},
  author={D{\'e}fossez, Alexandre and Adi, Yossi and Synnaeve, Gabriel},
  journal={TMLR},
  year={2022}
}
```

## License

This repository is released under the CC-BY-NC 4.0. license as found in the
[LICENSE](LICENSE) file, except for the following parts that is under the MIT license.
The files `examples/cifar/src/mobilenet.py` and `examples/cifar/src/src/resnet.py` are taken from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), released as MIT.
The file `examples/cifar/src/wide_resnet.py` is taken from [meliketoy/wide-resnet](https://github.com/meliketoy/wide-resnet.pytorch), released as MIT. See each file headers for the detailed license.

[api]: https://facebookresearch.github.io/diffq/diffq/index.html
[deit]: https://github.com/facebookresearch/deit
[fairseq]: https://github.com/pytorch/fairseq
[paper]: https://arxiv.org/abs/2104.09987
