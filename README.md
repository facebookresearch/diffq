# Differentiable Quantization


## To the reviewer

You will find in the `diffq` folder our generic framework for Differentiable Quantization
with pseudo quantization noise. In the `examples/` folder, you will find two applications,
to CIFAR-10/100 and visual transformer training. The `diffq` package will be found
automatically from the parent directory.
However you will still need to install the required dependencies for each examples, using
`pip install -r requirements.txt` from their respective subfolders.


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

To learn more on the API, open the file `docs/diffq/index.html` in your browser.


## Installation for development

This will install the dependencies and a `diffq` in developer mode (changes to the files
will directly reflect), along with the dependencies to run unit tests.
```
pip install -e '.[dev]'
```


## Test

You can run the unit tests with
```
make tests
```

## License

See the `LICENSE` file. This code is provided only for the purpose
of reviewing for the ICML 2021 conference. Any other use is prohibited.