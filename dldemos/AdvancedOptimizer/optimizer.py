import abc
from copy import deepcopy
from typing import Callable, Dict

import numpy as np


def const_lr(learning_rate_zero: float, epoch: int) -> float:
    return learning_rate_zero


def get_hyperbola_func(decay_rate: float) -> Callable[[float, int], float]:

    def scheduler(learning_rate_zero: float, epoch: int):
        return learning_rate_zero / (1 + epoch * decay_rate)

    return scheduler


class BaseOptimizer(metaclass=abc.ABCMeta):

    def __init__(
            self,
            param_dict: Dict[str, np.ndarray],
            learning_rate: float,
            lr_scheduler: Callable[[float, int], float] = const_lr) -> None:
        self.param_dict = param_dict
        self._epoch = 0
        self._num_step = 0
        self._learning_rate_zero = learning_rate
        self._lr_scheduler = lr_scheduler

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def learning_rate(self) -> float:
        return self._lr_scheduler(self._learning_rate_zero, self.epoch)

    def increase_epoch(self):
        self._epoch += 1

    def save(self) -> Dict:
        return {'epoch': self._epoch, 'num_step': self._num_step}

    def load(self, state_dict: Dict):
        self._epoch = state_dict['epoch']
        self._num_step = state_dict['num_step']

    def zero_grad(self):
        for k in self.grad_dict:
            self.grad_dict[k] = 0

    def add_grad(self, grad_dict: Dict[str, np.ndarray]):
        for k in self.grad_dict:
            self.grad_dict[k] += grad_dict[k]

    @abc.abstractmethod
    def step(self):
        pass


class GradientDescent(BaseOptimizer):

    def __init__(self, param_dict: Dict[str, np.ndarray],
                 learning_rate: float) -> None:
        super().__init__(param_dict, learning_rate)
        self.grad_dict = deepcopy(self.param_dict)

    def save(self) -> Dict:
        return super().save()

    def load(self, state_dict: Dict):
        super().load(state_dict)

    def step(self):
        self._num_step += 1
        for k in self.param_dict:
            self.param_dict[k] -= self.learning_rate * self.grad_dict[k]


class Momentum(BaseOptimizer):

    def __init__(self,
                 param_dict: Dict[str, np.ndarray],
                 learning_rate: float,
                 beta: float = 0.9,
                 from_scratch=False) -> None:
        super().__init__(param_dict, learning_rate)
        self.beta = beta
        self.grad_dict = deepcopy(self.param_dict)
        if from_scratch:
            self.velocity_dict = deepcopy(self.param_dict)
            for k in self.velocity_dict:
                self.velocity_dict[k] = 0

    def save(self) -> Dict:
        state_dict = super().save()
        state_dict['velocity_dict'] = self.velocity_dict
        return state_dict

    def load(self, state_dict: Dict):
        self.velocity_dict = state_dict.get('velocity_dict', None)
        if self.velocity_dict is None:
            self.velocity_dict = deepcopy(self.param_dict)
            for k in self.velocity_dict:
                self.velocity_dict[k] = 0
        super().load(state_dict)

    def step(self):
        self._num_step += 1
        for k in self.param_dict:
            self.velocity_dict[k] = self.beta * self.velocity_dict[k] + \
                (1 - self.beta) * self.grad_dict[k]
            self.param_dict[k] -= self.learning_rate * self.velocity_dict[k]


class RMSProp(BaseOptimizer):

    def __init__(self,
                 param_dict: Dict[str, np.ndarray],
                 learning_rate: float,
                 beta: float = 0.9,
                 eps: float = 1e-6,
                 from_scratch=False,
                 correct_param=True) -> None:
        super().__init__(param_dict, learning_rate)
        self.beta = beta
        self.eps = eps
        self.grad_dict = deepcopy(self.param_dict)
        self.correct_param = correct_param
        if from_scratch:
            self.s_dict = deepcopy(self.param_dict)
            for k in self.s_dict:
                self.s_dict[k] = 0

    def save(self) -> Dict:
        state_dict = super().save()
        state_dict['s_dict'] = self.s_dict
        return state_dict

    def load(self, state_dict: Dict):
        self.s_dict = state_dict.get('s_dict', None)
        if self.s_dict is None:
            self.s_dict = deepcopy(self.param_dict)
            for k in self.s_dict:
                self.s_dict[k] = 0
        super().load(state_dict)

    def step(self):
        self._num_step += 1
        for k in self.param_dict:
            self.s_dict[k] = self.beta * self.s_dict[k] + \
                (1 - self.beta) * np.square(self.grad_dict[k])
            if self.correct_param:
                s = self.s_dict[k] / (1 - self.beta**self._num_step)
            else:
                s = self.s_dict[k]
            self.param_dict[k] -= self.learning_rate * self.grad_dict[k] / (
                np.sqrt(s + self.eps))


class Adam(BaseOptimizer):

    def __init__(
            self,
            param_dict: Dict[str, np.ndarray],
            learning_rate: float,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
            from_scratch=False,
            correct_param=True,
            lr_scheduler: Callable[[float, int], float] = const_lr) -> None:
        super().__init__(param_dict, learning_rate, lr_scheduler)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_dict = deepcopy(self.param_dict)
        self.correct_param = correct_param
        if from_scratch:
            self.v_dict = deepcopy(self.param_dict)
            self.s_dict = deepcopy(self.param_dict)
            for k in self.v_dict:
                self.v_dict[k] = 0
                self.s_dict[k] = 0

    def save(self) -> Dict:
        state_dict = super().save()
        state_dict['v_dict'] = self.v_dict
        state_dict['s_dict'] = self.s_dict
        return state_dict

    def load(self, state_dict: Dict):
        self.v_dict = state_dict.get('v_dict', None)
        self.s_dict = state_dict.get('s_dict', None)
        if self.v_dict is None:
            self.v_dict = deepcopy(self.param_dict)
            for k in self.v_dict:
                self.v_dict[k] = 0
        if self.s_dict is None:
            self.s_dict = deepcopy(self.param_dict)
            for k in self.s_dict:
                self.s_dict[k] = 0
        super().load(state_dict)

    def step(self):
        self._num_step += 1
        for k in self.param_dict:
            self.v_dict[k] = self.beta1 * self.v_dict[k] + \
                (1 - self.beta1) * self.grad_dict[k]
            self.s_dict[k] = self.beta2 * self.s_dict[k] + \
                (1 - self.beta2) * (self.grad_dict[k] ** 2)
            if self.correct_param:
                v = self.v_dict[k] / (1 - self.beta1**self._num_step)
                s = self.s_dict[k] / (1 - self.beta2**self._num_step)
            else:
                v = self.v_dict[k]
                s = self.s_dict[k]
            self.param_dict[k] -= self.learning_rate * v / (np.sqrt(s) +
                                                            self.eps)
