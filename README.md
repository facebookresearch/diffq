# Differentiable Model Compression via Pseudo Quantization Noise
![linter badge](https://github.com/facebookresearch/diffq/workflows/linter/badge.svg)
![tests badge](https://github.com/facebookresearch/diffq/workflows/tests/badge.svg)
![cov badge](https://github.com/facebookresearch/diffq/workflows/cov%3E90%25/badge.svg)

DiffQ performs differentiable quantization using pseudo quantization noise.
It can automatically tune the number of bits used per weight or group of weights,
in order to achieve a given trade-off between model size and accuracy.

Go read [our paper][paper] for more details.

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
from diffq import DiffQuantizer

my_model = MyModel()
my_optim = ...  # The optimizer must be created before the quantizer
quantizer = DiffQuantizer(my_model)
quantizer.setup_optimizer(my_optim)

# Or, if you want to use a specific optimizer for DiffQ
quantizer.opt = torch.optim.Adam([{"params": []}])
quantizer.setup_optimizer(quantizer.opt)

# Distributed data parallel must be created after DiffQuantizer!
dmodel = torch.distributed.DistributedDataParallel(...)

# Then go on training as usual, just don't forget to call my_model.train() and my_model.eval().
penalty = 1e-3
for batch in loader:
    ...
    my_optim.zero_grad()
    # If you used a separate optimizer for DiffQ, call
    # quantizer.opt.zero_grad()

    # The `penalty` parameter here will control the tradeoff between model size and model accuracy.
    loss = F.mse_loss(x, y) + penalty * quantizer.model_size()
    my_optim.step()
    # If you used a separate optimizer for DiffQ, call
    # quantizer.opt.step()

# To get the true "naive" model size call
quantizer.true_model_size()

# To get the gzipped model size without actually dumping to disk
quantizer.compressed_model_size()

# When you want to dump your final model:
torch.save(quantizer.get_quantized_state(), "some_file.th")
# DiffQ will not optimally code integers. In order to actually get most
# of the gain in terms of size, you should call call `gzip some_file.th`.

# You can later load back the model with
quantizer.restore_quantized_state(torch.load("some_file.th"))
```

## Documentation

See the [API documentation][api].

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
  journal={arXiv preprint arXiv:2104.09987},
  year={2021}
}
```

## License

This repository is released under the CC-BY-NC 4.0. license as found in the
[LICENSE](LICENSE) file, except for the following parts that is under the MIT license.
The files `examples/cifar/src/mobilenet.py` and `examples/cifar/src/src/resnet.py` are taken from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), released as MIT.
The file `examples/cifar/src/wide_resnet.py` is taken from [meliketoy/wide-resnet](https://github.com/meliketoy/wide-resnet.pytorch), released as MIT. See each file headers for the detailed license.

[api]: https://facebookresearch.github.io/diffq/docs/diffq/index.html
[deit]: https://github.com/facebookresearch/deit
[fairseq]: https://github.com/pytorch/fairseq
[paper]: https://arxiv.org/abs/2104.09987
