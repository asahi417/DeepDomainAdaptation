import numpy as np


class StepScheduler:
    """ step size scheduler """

    def __init__(self,
                 current_epoch: int,
                 initial_step: float=None,
                 multiplier: float=1.0,
                 power: float=1.0,
                 exponential: bool=False,
                 identity: bool=False
                 ):

        if not exponential and initial_step is None:
            raise ValueError('initial step is needed')
        self.__initial_st = initial_step
        self.__current_ep = current_epoch
        self.__multiplier = multiplier
        self.__power = power
        self.__exponential = exponential
        self.__identity = identity

    def __call__(self):
        self.__current_ep += 1

        if self.__identity:
            return self.__initial_st
        elif self.__exponential:
            new_step = 2 / (1 + np.exp(-self.__multiplier * self.__current_ep)) - 1
            return new_step
        else:
            new_step = self.__initial_st / (1 + self.__multiplier * self.__current_ep) ** self.__power
            return new_step

