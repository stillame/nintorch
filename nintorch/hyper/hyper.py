"""Collection of scripts related to hyper parameter tuning.
"""
import optuna


__all__ = [
    'start_tuning',
    'default_search_neural_network']


def start_tuning(
    objective, direction: str = 'maximize', num_trials: int = 100):
    """
    """
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=num_trials)
    return study


def default_trial(trial):
    """Default hyper parameter search range for neural networks.
    In this case, weight decay and learning rate.
    """
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
    return {'weight_decay': weight_decay, 'lr': lr}



