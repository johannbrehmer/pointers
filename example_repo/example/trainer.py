import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import logging
from torch import nn
import copy

from example import utils

logger = logging.getLogger(__name__)


class Trainer(object):
    """ Trainer class. Any subclass has to implement the forward_pass() function. """

    def __init__(self, model, dtype=torch.float, device=torch.device("cpu"), n_workers=8, step_offset=0):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.n_workers = n_workers
        self.step_offset = step_offset

        self.train_dataset = self.val_dataset = None
        self.optim = self.sched = None
        self.train_loader = self.val_loader = None
        self._best_loss = float("inf")
        self._best_step = self._best_weights = None

    def train(
        self,
        train_dataset,
        val_dataset,
        steps,
        batchsize=100,
        lr=0.001,
        lr_decay=0.01,
        callback=None,
        callback_every_steps=1000,
        validation_every_steps=1000,
        clip_gradients=None,
        **kwargs,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self._initialize_models()
        self._initialize_training(**kwargs)
        self._make_dataloaders(batchsize)
        self._make_optim_sched(lr, lr_decay, steps)

        train_info = {}

        tbar = trange(steps)
        for step in tbar:
            batch = next(self.train_loader)
            batch = self._prep_batch(batch)
            self.model.train()
            self.optim.zero_grad()

            loss, batch_info = self._forward_pass(*batch, **kwargs)

            loss.backward()
            self.optim.step()
            self._clip_gradients(clip_gradients)
            self.sched.step()

            train_info = self._add_dicts(train_info, batch_info)
            self._report(tbar, train_info)

            if step == steps - 1 or (step + 1) % validation_every_steps == 0:
                self._validate(step)

            if (step == 0 or step == steps - 1 or (step + 1) % callback_every_steps == 0) and callback is not None:
                callback(self, step + self.step_offset)

        self._load_best_parameters()
        self._wrap_up_training()

    def _prep_batch(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = (batch,)
        batch = [x.to(self.device, self.dtype) for x in batch]
        return batch

    def _initialize_models(self):
        self.model = self.model.to(self.device, self.dtype)

    def _get_learnable_parameters(self):
        return self.model.parameters()

    def _make_optim_sched(self, lr, lr_decay, steps):
        parameters = self._get_learnable_parameters()
        self.optim = torch.optim.Adam(parameters, lr=lr)
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=lr_decay ** (1.0 / steps))

    def _make_dataloaders(self, batchsize):
        self.train_loader = InfiniteLoader(
            num_epochs=None, dataset=self.train_dataset, shuffle=True, batch_size=batchsize
        )
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=batchsize)

    def _clip_gradients(self, threshold):
        if threshold is None:
            return

        torch.nn.utils.clip_grad_norm_(self._get_learnable_parameters(), threshold)

    def _add_dicts(self, old, new, alpha=0.1):
        if old is None or not old:
            combined = new
        else:
            combined = {key: (1.0 - alpha) * val + alpha * new[key] for key, val in old.items()}
        return combined

    def _validate(self, step, **kwargs):
        # Compute validation loss
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._prep_batch(batch)
                batchsize = len(batch[0])

                loss, _ = self._forward_pass(*batch, **kwargs)
                val_loss += loss * batchsize / len(self.val_dataset)

            val_loss = val_loss.item()

        # Check loss against current best loss
        if val_loss < self._best_loss:
            self._best_step = step
            self._best_loss = val_loss
            self._best_weights = copy.deepcopy(self.model.state_dict())
            logger.info(f"Step {step}: validation loss = {val_loss} (new best)")
        else:
            logger.info(f"Step {step}: validation loss = {val_loss}")

    def _load_best_parameters(self):
        logger.info(f"Loading best weights after step {self._best_step} with validation loss {self._best_loss}")
        self.model.load_state_dict(self._best_weights)

    def _initialize_training(self, **kwargs):
        """ To be implemented by subclasses """
        pass

    def _forward_pass(self, *data, **kwargs):
        """ To be implemented by subclasses """
        raise NotImplementedError

    def _report(self, tbar, train_info):
        """ To be implemented by subclasses """
        pass

    def _wrap_up_training(self, **kwargs):
        """ To be implemented by subclasses """
        pass


class FlowTrainer(Trainer):
    def __init__(self, *args, dimensions=None, **kwargs):
        self._dimensions = dimensions
        super().__init__(*args, **kwargs)

    def _get_learnable_parameters(self):
        return self.model.flow.parameters()

    def _forward_pass(self, *batch, **kwargs):
        x = batch[0]
        log_likelihood = self.model.flow.log_prob(x)
        loss = -torch.mean(log_likelihood)
        if self._dimensions is not None:
            loss = utils.nats_to_bits_per_dim(loss, self._dimensions)
        info = {"loss": loss.item()}
        return loss, info

    def _report(self, tbar, train_info):
        tbar.set_description(f"NLL = {train_info['loss']:.2f}]")
        tbar.refresh()


class ClassifierTrainer(Trainer):
    def _get_learnable_parameters(self):
        return self.model.classifier.parameters()

    def _initialize_training(self, **kwargs):
        self.criterion = nn.BCELoss()

    def _forward_pass(self, *batch, **kwargs):
        x, y = batch
        s = self.model.classifier(x)
        loss = self.criterion(s, y)

        info = {"loss": loss.item()}
        return loss, info

    def _report(self, tbar, train_info):
        tbar.set_description(f"XE = {train_info['loss']:.2f}")
        tbar.refresh()


class InfiniteLoader(DataLoader):
    """A data loader that can load a dataset repeatedly. From https://github.com/bayesiains/nsf """

    def __init__(self, num_epochs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finite_iterable = super().__iter__()
        self.counter = 0
        self.num_epochs = float("inf") if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = super().__iter__()
            return next(self.finite_iterable)

    def __iter__(self):
        return self

    def __len__(self):
        return None
