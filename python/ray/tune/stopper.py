class Stopper:
    """Base class for implementing a Tune experiment stopper.

    Allows users to implement experiment-level stopping via ``stop_all``. By
    default, this class does not stop any trials. Subclasses need to
    implement ``__call__`` and ``stop_all``.

    .. code-block:: python

        import time
        from ray import tune
        from ray.tune import Stopper

        class TimeStopper(Stopper):
            def __init__(self):
                self._start = time.time()
                self._deadline = 300

            def __call__(self, trial_id, result):
                return False

            def stop_all(self):
                return time.time() - self._start > self.deadline

        tune.run(Trainable, num_samples=200, stop=TimeStopper())

    """

    def __call__(self, trial_id, result):
        """Returns true if the trial should be terminated given the result."""
        raise NotImplementedError

    def stop_all(self):
        """Returns true if the experiment should be terminated."""
        raise NotImplementedError


class NoopStopper(Stopper):
    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return False


class FunctionStopper(Stopper):
    def __init__(self, function):
        self._fn = function

    def __call__(self, trial_id, result):
        return self._fn(trial_id, result)

    def stop_all(self):
        return False

    @classmethod
    def is_valid_function(cls, fn):
        is_function = callable(fn) and not issubclass(type(fn), Stopper)
        if is_function and hasattr(fn, "stop_all"):
            raise ValueError(
                "Stop object must be ray.tune.Stopper subclass to be detected "
                "correctly.")
        return is_function
    
    
class EarlyStopping(Stopper):
    """Early stops the training if validation performance doesn't improve after a given patience."""

    def __init__(self,
                 patience=7,
                 verbose=False,
                 delta=0,
                 monitor='val_loss',
                 mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Evaluation metric to monitor for the performance
            mode (str): min or max.
        """
        self.patience = patience
        self.verbose = verbose
        self.monitor = monitor
        self.mode = mode
        self.counter = {}
        self.best_score = {}
        self.early_stop = {}
        self.delta = delta

    def __call__(self, trial_id, result):
        """Returns true if the trial should be terminated given the result."""
        if self.mode == 'min':
            score = -result[self.monitor]
        else:
            score = result[self.monitor]

        if trial_id in self.best_score:
            if score <= self.best_score[trial_id] + self.delta:
                self.counter[trial_id] += 1
                print(f'EarlyStopping counter: {self.counter[trial_id]} out of {self.patience}')
                if self.counter[trial_id] >= self.patience:
                    self.early_stop[trial_id] = True
            else:
                self.best_score[trial_id] = score
                self.counter[trial_id] = 0
        else:
            self.counter[trial_id] = 0
            self.best_score[trial_id] = score
            self.early_stop[trial_id] = False

        return self.early_stop[trial_id]

    def stop_all(self):
        """Returns true if the experiment should be terminated."""
        return False
