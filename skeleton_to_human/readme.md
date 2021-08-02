This is the pytorch lightning version of [Transmomo](https://github.com/yzhq97/transmomo.pytorch). 

## Directory Structure

```sh
├── readme.md
├── skeleton_to_human
│   ├── data
│   │   ├── __init__.py
│   │   ├── aligned_pair_dataset.py
│   │   └── dancing.py
        └── image_folder.py
        └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   └── __init__.py
│   ├── models
│   │   ├── __init__.py
        ├── networks.py
│   │   └── pose2vidHD_model.py
    ├── util
│       └── util.py
└── training
    ├── __init__.py
    └── run_experiment_skeleton.py
```

`skeleton_to_image` is a the package that converts skeleton images to human images.

The `training` folder outside this directory is the support code for developing `skeleton_to_image`, which consists of `run_experiment.py`.

Within `skeleton_to_image`, there is further breakdown between `data`, `models`, and `lit_models`.


### Data

There are three scopes of our code dealing with data, with slightly overlapping names: `DataModule`, `DataLoader`, and `Dataset`.

At the top level are `DataModule` classes, which are responsible for quite a few things:

- Processing data as needed to get it ready to go through PyTorch models
- Splitting data into train/val/test sets
- Specifying dimensions of the inputs (e.g. `(C, H, W) float tensor`
- Specifying information about the targets (e.g. a class mapping)
- Specifying data augmentation transforms to apply in training

In the process of doing the above, `DataModule`s make use of a couple of other classes:

1. They wrap underlying data in a `torch Dataset`, which returns individual (and optionally, transformed) data instances.
2. They wrap the `torch Dataset` in a `torch DataLoader`, which samples batches, shuffles their order, and delivers them to the GPU.

### Models

The main model is defined here. It consists only bare minimum part of the whole architecture that is needed for inference. 

### Lit Models

I use PyTorch-Lightning for training, which defines the `LightningModule` interface that handles not only everything that a Model (as defined above) handles, but also specifies the details of the learning algorithm: what loss should be computed from the output of the model and the ground truth, which optimizer should be used, with what learning rate, etc.

## Training

`training/run_experiment.py` is a script that handles many command-line parameters.


```sh
python3 training/run_experiment.py <add arguments>
```

While `model_class` and `data_class` are our own arguments, `max_epochs` and `gpus` are arguments automatically picked up from `pytorch_lightning.Trainer`.
You can use any other `Trainer` flag (see [docs](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags)) on the command line, for example `--batch_size=512`.

The `run_experiment.py` script also picks up command-line flags from the model and data classes that are specified.
For example, in `skeleton_to_image/models/mlp.py` we specify the `MLP` class, and add a couple of command-line flags: `--fc1` and `--fc2`.

