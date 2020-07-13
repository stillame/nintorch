"""Collection of scripts related to hyper parameter tuning.
"""
from typing import Callable
import optuna
import gc
from .. import 


__all__ = [
    'start_tuning',
    'default_trial']


def start_tuning(
    objective, direction: str = 'maximize', num_trials: int = 100, simpler = None):
    """TODO: checking the default value of simpler.
    """
    if simpler is None:
        study = optuna.create_study(direction=direction)
    else:
        study = optuna.create_study(direction=direction, simpler=simpler)
    study.optimize(objective, n_trials=num_trials)
    return study


def default_trial(trial):
    """Default hyper parameter search range for neural networks.
    In this case, weight decay and learning rate.
    """
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    return {'weight_decay': weight_decay}


def objective(epoch: int, trainer, trial_func: Callable, trial) -> float:
    """Objective function for hyper parameter training.
    """
    acc = trainer.train_eval_epoches(epoch, 1, 0, True, trial_func, trial)
    # Setting for next iteration of training.
    del trainer
    gc.collect()
    return acc


