"""Collection of scripts related to hyper parameter tuning.
"""
from typing import Callable
import optuna
import gc


__all__ = [
    'start_tuning',
    'default_trial',
    'default_objective',
    ]


def start_tuning(
    objective, direction: str = 'maximize', n_trials: int = 10, simpler = None):
    """TODO: checking the default value of simpler.
    """
    assert direction in ['maximize', 'minimize']
    assert isinstance(n_trials, int)
    if simpler is None:
        study = optuna.create_study(direction=direction)
    else:
        study = optuna.create_study(direction=direction, simpler=simpler)
    study.optimize(objective, n_trials=n_trials)
    return study


def default_trial(trial):
    """Default hyper parameter search range for neural networks.
    In this case, weight decay and learning rate.
    """
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    momentum = trial.suggest_uniform('momentum', 0.0, 0.9)
    return {'weight_decay': weight_decay, 'lr': lr, 'momentum': momentum}


def default_objective(
        epoch: int, trainer, trial_func: Callable, trial: Callable) -> float:
    """Wrapped objective function for hyper parameter training.
    """
    assert isinstance(epoch, int)
    # Model and optim are required for reconstructing every trainning time.
    best_acc = trainer.train_eval_epoches(
        epoch, 1, 'test', trial_func, trial, True, 2)
    return best_acc
