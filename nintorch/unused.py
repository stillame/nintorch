"""Collection of functions which is considered to removed.
"""
import optuna
from nintorch import Trainer
from funtools import defaultdict
from copy import deepcopy


class HyperTrainer(Trainer):
    """With the solution too complex and problem with the unknown prunning
    situation. Will be removed this one.
    """
    def __init__(self, valid_or_test: bool = False, *args, **kwargs):
        super(HyperTrainer, self).__init__(*args, **kwargs)
        self.valid_or_test = valid_or_test
        self.init_model = deepcopy(self.model)

    def reset(self):
        """Reset setting within Trainer for every trial of training.
        Is this solution too complex? 
        """
        self.dfs = {}
        self.model = deepcopy(self.init_model)

        #self.optim.param_groups[0]['params'] = {}
        # From: https://discuss.pytorch.org/t/reset-adaptive-optimizer-state/14654
        # Reset adaptive variable to the default.
        self.optim.__setstate__({'state': defaultdict(dict)})
        self.optim.add_param_group({'params': self.model.parameters()})
        self.epoch_idx = 0

    def train_eval_epoches(
        self, epoch: int, eval_every_epoch: int,
        verbose: int, return_best: bool, trial_funct, trial) -> float:
        """Train and test for certain epoches with an option
        to turn on the hyper parameter tunning.
        For trial_funct, please follow nintorch.hyper.default_trial.

        Having problem with optuna, with class.method the constructor is not
        called for each trial of optuna. Solving by using self.init_model keep
        for reseting the model.
        """
        self._check_train()
        HEADER: str = 'train'
        self.check_df_exists(HEADER, ['acc', 'loss'])
        if self.valid_or_test:
            self._check_valid()
            HEADER_EVAL: str = 'validation'
        else:
            self._check_test()
            HEADER_EVAL: str = 'test'
        kwargs = trial_funct(trial)
        self.check_df_exists(HEADER_EVAL, ['acc', 'loss'])

        # TODO: cover more than lr, weight decay and momentum.
        if 'lr' in kwargs:
            self.optim.param_groups[0]['lr'] = kwargs['lr']
        if 'weight_decay' in kwargs:
            self.optim.param_groups[0]['weight_decay'] = kwargs['weight_decay']
        if 'momentum' in kwargs:
            self.optim.param_groups[0]['momentum'] = kwargs['momentum']
        if len(kwargs) > 3:
            warnings.warn('kwargs is more than 3 might not supported.', UserWarning)

        for i in range(epoch):
            train_acc, train_loss = self.train_an_epoch()
            
            if verbose > 0:
                self.log_info(
                   header=HEADER, epoch=self.epoch_idx,
                    acc=train_acc, loss=train_loss)
            self.dfs_append_row(HEADER, acc=train_acc, loss=train_loss)

            if i % eval_every_epoch == 0 and i != 0:
                if self.valid_or_test:
                    eval_acc, eval_loss = self.valid_an_epoch()
                else:
                    eval_acc, eval_loss = self.test_an_epoch()

                if verbose > 0:
                    self.log_info(
                       header=HEADER_EVAL, epoch=self.epoch_idx,
                        acc=eval_acc, loss=eval_loss)

                self.dfs_append_row(HEADER_EVAL, acc=eval_acc, loss=eval_loss)
                trial.report(eval_acc, i)
            if trial.should_prune():
                self.reset()
                raise optuna.exceptions.TrialPruned()

        if return_best:
            best_acc = self.get_best(HEADER_EVAL, 'acc', 'max')
            self.reset()
            return best_acc
        else:
            self.reset()
            return eval_acc


