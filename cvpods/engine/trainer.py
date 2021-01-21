# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by BaseDetection, Inc. and its affiliates. All Rights Reserved
import logging
import math
import os
import time
import weakref
from collections import OrderedDict

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.data import build_test_loader, build_train_loader
from cvpods.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results
)
from cvpods.modeling.nn_utils.module_converter import maybe_convert_module
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
    comm,
    setup_logger
)

from . import hooks
from .hooks import HookBase

try:
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )


__all__ = ["TrainerBase", "SimpleTrainer", "DefaultTrainer"]


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

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
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int, max_epoch):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        self.max_epoch = max_epoch

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, batch_subdivisions=1):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            batch_subdivisions: an integer. Batchsize must be divisible by `batch_subdivisions`
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.data_loader = data_loader
        self.model = model
        self.epoch = 0
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.batch_subdivisions = batch_subdivisions

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        start = time.perf_counter()
        data_time_sum = 0.
        loss_dict_summary = {}
        # for each mini step
        for division_iter in range(self.batch_subdivisions):
            start = time.perf_counter()
            try:
                data = next(self._data_loader_iter)
            except StopIteration:
                # start new epoch
                self.epoch += 1
                self.data_loader.sampler.set_epoch(self.epoch)
                self._data_loader_iter = iter(self.data_loader)
                data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start
            data_time_sum += data_time

            """
            If your want to do something with the losses, you can wrap the model.
            """
            loss_dict = self.model(data)

            for metrics_name, metrics_value in loss_dict.items():
                # Actually, some metrics are not loss, such as
                # top1_acc, top5_acc in classification, filter them out
                if metrics_value.requires_grad:
                    loss_dict[metrics_name] = metrics_value / self.batch_subdivisions

            losses = sum([
                metrics_value for metrics_value in loss_dict.values()
                if metrics_value.requires_grad
            ])
            self._detect_anomaly(losses, loss_dict)

            # if use fp16 mode
            if self.cfg.TRAINER.FP16.ENABLED:
                # only in last subdivision iter, DDP needs to backward with sync
                if (
                    division_iter != self.batch_subdivisions - 1
                    and isinstance(self.model, DistributedDataParallel)
                ):
                    with self.model.no_sync():
                        with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                else:
                    with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
            else:
                if (
                    division_iter != self.batch_subdivisions - 1
                    and isinstance(self.model, DistributedDataParallel)
                ):
                    with self.model.no_sync():
                        losses.backward()
                else:
                    losses.backward()

            # The values in dict: `loss_dict` can be divided into two cases:
            #   * case 1. value.requires_grad = True, this values is loss, need to be summed
            #   * case 2. value.requires_grad = False, like top1_acc, top5_acc in classification ...
            #         use the last mini_step value as the current iter value.
            for metrics_name, metrics_value in loss_dict.items():
                if metrics_name not in loss_dict_summary:
                    loss_dict_summary[metrics_name] = metrics_value
                elif metrics_value.requires_grad:
                    loss_dict_summary[metrics_name] += metrics_value  # Sum the loss
                else:
                    loss_dict_summary[metrics_name] = metrics_value  # Update other metrics

        metrics_dict = {"data_time": data_time_sum}
        metrics_dict.update(loss_dict_summary)
        self._write_metrics(metrics_dict)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in cvpods.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for key, loss in metrics_dict.items() if "loss" in key)

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


class DefaultTrainer(SimpleTrainer):
    """
    A trainer with default training logic. Compared to `SimpleTrainer`, it
    contains the following logic in addition:

    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it mades.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in cvpods.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Attributes:
        scheduler:
        checkpointer (DefaultCheckpointer):
        cfg (BaseConfig):

    Examples:
    .. code-block:: python

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    """
    def __init__(self, cfg, model_build_func):
        """
        Args:
            cfg (BaseConfig):
        """
        logger = logging.getLogger("cvpods")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        self.start_iter = 0

        data_loader = self.build_train_loader(cfg)
        maybe_adjust_epoch_and_iter(cfg, data_loader)
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH

        model = model_build_func(cfg)
        model = maybe_convert_module(model)
        logger.info(f"Model structure: {model}")

        # Assume these objects must be constructed in this order.
        optimizer = self.build_optimizer(cfg, model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            if cfg.TRAINER.FP16.ENABLED:
                if cfg.TRAINER.FP16.TYPE == "APEX":
                    model, optimizer = amp.initialize(
                        model, optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                    )
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True
            )
        # TODO: @wangfeng02, `batch_subdivisions`
        super().__init__(model, data_loader, optimizer, cfg.SOLVER.BATCH_SUBDIVISIONS)

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        self.scheduler = self.build_lr_scheduler(
            cfg, optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        optional = {}
        if cfg.TRAINER.FP16.ENABLED:
            optional["amp"] = amp
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
            **optional,
        )

        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume = resume
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

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
            ) if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD)
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=self.cfg.GLOBAL.LOG_INTERVAL
            ))
            # Put `PeriodicDumpLog` after writers so that can dump all the files,
            # including the files generated by writers
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
            .. code-block:: python

                return [
                    CommonMetricPrinter(self.max_iter),
                    JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                    TensorboardXWriter(self.cfg.OUTPUT_DIR),
                    ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter, self.max_epoch)

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(self, "_last_eval_results"
                           ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
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
        Overwrite it if you'd like a different data loader.:w
        """
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_test_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            "Please either implement `build_evaluator()` in subclasses, or pass "
            "your evaluator as arguments to `DefaultTrainer.test()`.")

    @classmethod
    def test(cls, cfg, model, evaluators=None, output_folder=None):
        """
        Args:
            cfg (BaseConfig):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(
                cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                    len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(
                        cfg, dataset_name, data_loader.dataset, output_folder=output_folder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i)
                logger.info("Evaluation results for {} in csv format:".format(
                    dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def maybe_adjust_epoch_and_iter(cfg, dataloader):
    logger = logging.getLogger(__name__)

    max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
    max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER

    subdivision = cfg.SOLVER.BATCH_SUBDIVISIONS
    # adjust lr by batch_subdivisions
    cfg.SOLVER.OPTIMIZER.BASE_LR *= subdivision

    if max_epoch:
        epoch_iter = math.ceil(len(dataloader.dataset) /
                               (cfg.SOLVER.IMS_PER_BATCH * subdivision))

        if max_iter is not None:
            logger.warning(
                f"Training in EPOCH mode, automatically convert {max_epoch} epochs "
                f"into {max_epoch*epoch_iter} iters..."
            )

        cfg.SOLVER.LR_SCHEDULER.MAX_ITER = max_epoch * epoch_iter
        cfg.SOLVER.LR_SCHEDULER.STEPS = [x * epoch_iter for x in cfg.SOLVER.LR_SCHEDULER.STEPS]
        cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS = int(
            cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS * epoch_iter
        )
        cfg.SOLVER.CHECKPOINT_PERIOD = epoch_iter * cfg.SOLVER.CHECKPOINT_PERIOD
        cfg.TEST.EVAL_PERIOD = epoch_iter * cfg.TEST.EVAL_PERIOD
    else:
        epoch_iter = -1

    cfg.SOLVER.LR_SCHEDULER.EPOCH_ITERS = epoch_iter
