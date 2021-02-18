# Differentiable Quantization

DiffQ performs differentiable quantization using pseudo quantization noise.
It can automatically tune the number of bits used per weight or group of weight,
in order to achieve a given trade-off between the model size and accuracy.

## Requirements

DiffQ requires only a reasonably recent version of PyTorch.
To install DiffQ, you can run from the root of the repository:

```
pip install .
```

You must do so before trying one of the examples.

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
quantizer.setup_opt(quantizer.opt)

# Distributed data parallel must be created after DiffQuantizer!

dmodel = torch.distributed.DistributedDataParallel(...)

# Then go on training as usual, just don't forget to call my_model.train() and my_model.eval().
...
mu = 1e-3
for batch in loader:
    ...
    my_optim.zero_grad()
    # If you used a separate optimizer for DiffQ, call
    # quantizer.opt.zero_grad()

    # The `mu` parameter here will control the tradeoff between model size
    # and model accuracy.
    loss = F.mse_loss(x, y) + mu * quantizer.model_size()
    my_optim.step()
    # If you used a separate optimizer for DiffQ, call
    # quantizer.opt.step()

# To get the true "naive" model size call
quantizer.true_model_size()

# To get the gzipped model size without actually dumping to disk
quantizer.compressed_model_size()

# When you want to dump your final model:
torch.save(quantizer.get_quantized_state(), "some_file.th")
# And then call `gzip some_file.th`, to get the actual quantized size.
# You can later load back the model with
quantizer.restore_quantized_state(torch.load("some_file.th"))
```

## Documentation

See the [API documentation][api].

## Examples

We provide two examples in the `examples/` folder. One is for CIFAR-10/100,
using standard architecture such as Wide-ResNet, ResNet or MobileNet.
The second is based on the [DeiT][deit] visual transformer.

The DeiT example is provided as a patch on the original codebase at a specific
commit. You can initialize the git submodule and apply the patch by running

```
examples/setup_deit.sh
```


## Installation for development

This will install the dependencies and a `diffq` in developer mode (changes to the files
will directly reflect), along with the dependencies to run unit tests.
```
pip install -e '.[dev]'
```


### Test

You can run the unit tests with
```
make tests
```

## License

This repository is released under the CC-BY-NC 4.0. license as found in the
[LICENSE](LICENSE) file.


[api]: https://share.honu.io/diffq/docs/diffq/index.html
[deit]: https://github.com/facebookresearch/deit