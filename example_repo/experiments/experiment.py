#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")

import sys
import os
import warnings
import logging
from pathlib import Path
import numpy as np
import torch

from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
SETTINGS["CAPTURE_MODE"] = "sys"  # Fixing a bug in sacred when used with multiprocessing

sys.path.insert(0, "../")
from experiments.datasets import get_dataset, get_dataset_resolution
from example import FlowTrainer

logger = logging.getLogger(__name__)
ex = Experiment(name="crow")
ex.logger = logger


# noinspection PyUnusedLocal
@ex.config
def config():
    # Main settings
    name = "model"
    dataset = "imagenet"
    seed = 12345
    run_name = f"{dataset}_{name}_{seed}"

    # Task
    mode = "all"

    # Hyperparameters
    batchsize = 100
    lr = 1.e-3
    lr_decay = 1.e-2
    validation_frequency = 1000
    callback_frequency = 1000

    # Filenames
    base = Path(".")
    data_dir = base / "data"
    dataset_dir = data_dir / "datasets" / dataset
    run_dir = data_dir / "runs" / run_name
    model_filename = run_dir / f"{run_name}.pty"
    model_snapshot_filename = run_dir / "model_snapshots" / f"{run_name}_step_{{}}.pty"
    plot_snapshot_filename = run_dir / "training_progress" / f"{run_name}_step_{{}}.pdf"
    plot_filename = run_dir / f"{run_name}.pdf"
    results_filename = run_dir / f"{run_name}.npz"
    file_observer_filename = run_dir

    # Technical stuff
    device = torch.device("cpu")
    gpu = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = True
    dtype = torch.float
    debug = False

    # Observers
    ex.observers.append(FileStorageObserver(file_observer_filename))


@ex.named_config
def config1():
    dataset = "fashion-mnist"


@ex.named_config
def config2():
    dataset = "imagenet"


@ex.capture
def setup_logging(debug, silence_list=("matplotlib",)):
    """ Sets up logging  """

    # Experiment logger
    logger.handlers = []
    ch = logging.StreamHandler(sys.stdout)
    if debug:
        formatter = logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%y-%m-%d %H:%M")
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname).1s: %(message)s", datefmt="%y-%m-%d %H:%M")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Set levels of all the loggers out there
    for key in logging.Logger.manager.loggerDict.keys():
        that_logger = logging.getLogger(key)
        that_logger.setLevel(logging.DEBUG if debug else logging.INFO)
        for check_key in silence_list:
            if check_key in key:
                that_logger.setLevel(logging.ERROR)
                break

    # Ignore warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")


@ex.capture
def setup_run(
    name,
    run_name,
    mode,
    seed,
    model_filename,
    results_filename,
    model_snapshot_filename,
    dataset_image_dir,
    gpu,
):
    """ Sets up run, including the random seed  """

    logger.info(f"Setting up run {name} ({run_name}) in mode {mode}")

    os.makedirs(dataset_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    os.makedirs(os.path.dirname(model_snapshot_filename), exist_ok=True)

    logger.info(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    if gpu:
        logger.info("Running on GPU")
    else:
        logger.info("Running on CPU")

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)


@ex.capture
def create_model():
    raise NotImplementedError


@ex.capture
def load_model(model, model_filename):
    logger.info(f"Loading model state from {model_filename}")
    model.load_state_dict(torch.load(model_filename, map_location=torch.device("cpu")))


@ex.capture
def callback(trainer, step, model_snapshot_filename):
    save_model(trainer.model, model_filename=str(model_snapshot_filename).format(step))


@ex.capture
def train(model, dataset, dataset_dir, batchsize, steps, lr, lr_decay, callback_frequency, validation_frequency, validation_fraction, gpu, dtype, device):
    # Get dataset
    chw = get_dataset_resolution(dataset, dataset_dir)
    train_set, val_set = get_dataset(dataset, directory=dataset_dir, partition="train_val", valid_fraction=validation_fraction)

    # This has to happen after getting the datasets (which for now live on the CPU memory):
    if gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Train model
    trainer = FlowTrainer(dtype=dtype, device=device, model=model, dimensions=chw)
    trainer.train(
        train_set,
        val_set,
        batchsize=batchsize,
        lr=lr,
        lr_decay=lr_decay,
        steps=steps,
        callback=callback,
        callback_every_steps=callback_frequency,
        validation_every_steps=validation_frequency,
    )


@ex.capture
def save_model(model, model_filename, add_artifact=False):
    logger.debug(f"Saving model at {model_filename}")
    torch.save(model.state_dict(), model_filename)
    if add_artifact:
        ex.add_artifact(model_filename)


@ex.capture
def evaluate(model):
    raise NotImplementedError


@ex.automain
def main():
    """ Main entry point for experiments """

    setup_logging()
    logger.info("Hi!")
    setup_run()

    model = create_model()

    train(model)
    save_model(model, add_artifact=True)
    result = evaluate(model)

    logger.info("That's all, have a nice day!")
    return result
