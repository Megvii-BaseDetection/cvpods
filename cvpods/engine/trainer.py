import argparse
import logging
import os
import sys
import time
import weakref
from collections import OrderedDict
from typing import Dict, Optional
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules

import numpy as np

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.data import (
    build_test_loader,
    build_train_loader
)
from cvpods.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results
)
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils import comm
from cvpods.utils.env import TORCH_VERSION, seed_all_rng
from cvpods.utils.dump.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)
from cvpods.utils import setup_logger
from cvpods.utils.registry import Registry
from cvpods.modeling.nn_utils.module_converter import maybe_convert_module

from .hooks import HookBase
from . import hooks


RUNNERS = Registry("runners")


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


@RUNNERS.register()
class RunnerBase:
    """
    Base class for iterative runner with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the runner. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and runner cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage._iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == runner.iter
        # for the entire execution of each step
        self.storage._iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError


@RUNNERS.register()
class SimpleRunner(RunnerBase):
    """
    A simple runner for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`RunnerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass RunnerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the runner.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleRunner] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._epoch += 1
            if hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(self._epoch)
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward
        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict["data_time"] = data_time

            # Gather metrics among all workers for logging
            # This assumes we do DDP-style training, which is currently the only
            # supported method in cvpods.
            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)


@RUNNERS.register()
class AMPRunner(SimpleRunner):
    """
    Like :class:`SimpleRunner`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleRunner`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPRunner does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPRunner] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPRunner] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


@RUNNERS.register()
class DefaultRunner(RunnerBase):
    """
    A runner with default training logic. It does the following:

    1. Create a :class:`SimpleRunner` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleRunner` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleRunner`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in cvpods.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        runner = DefaultRunner(cfg)
        runner.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        runner.train()

    Attributes:
        scheduler:
        checkpointer (DefaultCheckpointer):
        cfg (config dict):
    """

    def __init__(self, cfg, build_model):
        """
        Args:
            cfg (config dict):
        """
        super().__init__()
        logger = logging.getLogger("cvpods")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for cvpods 
            setup_logger()
        self.logger = logger

        self.data_loader = self.build_train_loader(cfg)
        # Assume these objects must be constructed in this order.
        model = build_model(cfg)
        self.model = maybe_convert_module(model)
        self.logger.info(f"Model: \n{self.model}")

        # Assume these objects must be constructed in this order.
        self.optimizer = self.build_optimizer(cfg, self.model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True)

        self._runner = (AMPRunner if cfg.TRAINER.FP16.ENABLED else SimpleRunner)(
            self.model, self.data_loader, self.optimizer
        )

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            self.logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        self.udpate_iter_by_epoch(cfg, self.data_loader)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your runner.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._runner.iter = self.iter
        self._runner.run_step()

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`cvpods.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, **kwargs):
        """
        It now calls :func:`cvpods.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer, **kwargs)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            """
            If you want DefaultRunner to automatically run evaluation,
            please implement `build_evaluator()` in subclasses (see train_net.py for example).
            Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
            """
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (config dict):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultRunner.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def udpate_iter_by_epoch(self, cfg, dataloader):

        max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER

        if max_epoch:
            epoch_iter = math.ceil(
                len(dataloader.dataset) / cfg.SOLVER.IMS_PER_BATCH)

            if max_iter is not None:
                self.logger.warning(
                    f"Training in EPOCH mode, automatically convert {max_epoch} epochs "
                    f"into {max_epoch*epoch_iter} iters...")

            cfg.SOLVER.LR_SCHEDULER.MAX_ITER = max_epoch * epoch_iter
            cfg.SOLVER.LR_SCHEDULER.STEPS = [
                x * epoch_iter for x in cfg.SOLVER.LR_SCHEDULER.STEPS
            ]
            cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS = int(
                cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS * epoch_iter)
            cfg.SOLVER.CHECKPOINT_PERIOD = epoch_iter * cfg.SOLVER.CHECKPOINT_PERIOD
            cfg.TEST.EVAL_PERIOD = epoch_iter * cfg.TEST.EVAL_PERIOD
        else:
            epoch_iter = -1

        cfg.SOLVER.LR_SCHEDULER.EPOCH_ITERS = epoch_iter
