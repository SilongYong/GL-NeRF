# Running Vanilla NeRF with the Gauss Laguerre Quadrature

## Installation

Since this code is based on [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch/tree/master), please follow their installation guide. In short,

```shell
pip install -r requirements.txt
```



## Training
For NeRF-Synthetic dataset, run
```shell
python run_nerf.py --expname <expname> --config-file /path/to/configs/<scene>.yaml --opts task train N_samples 64 N_importance 128
```

For LLFF dataset, run
```shell
python run_nerf.py --expname <expname> --config-file /path/to/configs/<scene>.yaml --opts task train N_samples 64 N_importance 64
```

Here `<expname>` is the name for your experiment and `<scene>` refers to the name of the scene you plan to train on.

## Evaluation
- Evaluation of trained NeRF models with different inference method:

  Vanilla:

  ```shell
  python run_nerf.py --expname <expname> --config-file /path/to/configs/<scene>.yaml --opts task test N_samples 128 N_importance 64
  ```

  Gauss-Laguerre Quadrature:

  ```shell
  python run_nerf.py --expname <expname> --config-file /path/to/configs/<scene>.yaml --opts task test_lag N_samples 128 N_importance 16
  ```

  Note we can also reduce the number of sample points used for Gauss-Laguerre quadrature and the performance drop is marginal. `16` is the number for producing the paper results regarding vanilla NeRF

## Acknowledgements
We would like to thank [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch/tree/master) for the useful code base.