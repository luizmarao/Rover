from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnNoModelImprovement
import numpy as np

class RoverCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    - Reset the environments when starting a rollout, to ensure proper measurement of success and failure rates.
    - Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.
    - It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """
    def __init__(self, max_no_improvement_evals: int = 1000, min_evals: int = 0, verbose: int = 0):
        super(RoverCallback, self).__init__(verbose)
        #TODO: implement stop when the model is not improving -> see StopTrainingOnNoModelImprovement(BaseCallback):

        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0
        self.continue_training = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.training_env.reset()
        if self.verbose >=2 :
            print("The environment has been reseted")


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
