from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def quadratic_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Quadratic learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return (progress_remaining ** 2) * initial_value

    return func

def exponential_schedule(initial_value: float, decay_rate=0.9995,
                         num_timesteps=1.2e8, n_steps = 8192) -> Callable[[float], float]:
    """
    Exponential learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        progress = 1 - progress_remaining
        steps_so_far = progress * num_timesteps
        return initial_value * decay_rate ** (steps_so_far/n_steps)

    return func