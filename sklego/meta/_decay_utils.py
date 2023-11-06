import numpy as np


class LinearDecay:
    """Class representing a linear decay.

    This class generates a linear decay function that maps input data `X`, `y` to a linearly decreasing range from
    `max_value` to `min_value`. The length and step of the decay is determined by the number of samples in `y`.

    !!! warning
        It is up to the user to sort the dataset appropriately.

    Parameters
    ----------
    min_value : float, default=0.
        The minimum value of the decay.
    max_value : float, default=1.
        The maximum value of the decay.

    Raises
    ------
    ValueError
        If `min_value` is greater than `max_value`.
    """

    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, X, y):
        """Compute the decay values based on the shape of `y`.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features,)
            Training data. Unused, present for API consistency by convention.
        y : array-like, shape=(n_samples,)
            Target values. Used to determine the number of samples in the decay.

        Returns
        -------
        np.ndarray, shape=(n_samples,)
            The decay values.
        """
        if self.min_value > self.max_value:
            raise ValueError("`min_value` must be less than or equal to `max_value`")

        n_samples = y.shape[0]
        return np.linspace(self.min_value, self.max_value, n_samples)


class ExponentialDecay:
    r"""Class representing an exponential decay.

    This class generates an exponential decay function that maps input data `X`, `y` to a exponential decreasing range
    $w_{t-1} = decay\_rate * w_{t}$. The length of the decay is determined by the number of samples in `y`.

    !!! warning
        It is up to the user to sort the dataset appropriately.

    Parameters
    ----------
    decay_rate : float, default=0.999
        The rate of decay.

    Raises
    ------
    ValueError
        If `decay_rate` not between 0 and 1.
    """

    def __init__(self, decay_rate=0.999):
        self.decay_rate = decay_rate

    def __call__(self, X, y):
        """Compute the decay values based on the shape of `y`.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features,)
            Training data. Unused, present for API consistency by convention.
        y : array-like, shape=(n_samples,)
            Target values. Used to determine the number of samples in the decay.

        Returns
        -------
        np.ndarray, shape=(n_samples,)
            The decay values.
        """
        if self.decay_rate <= 0 or self.decay_rate >= 1:
            raise ValueError(
                f"`decay_rate` must be between 0. and 1., found {self.decay_rate}"
            )

        n_samples = y.shape[0]
        return self.decay_rate ** np.arange(n_samples, 0, -1)


class StepWiseDecay:
    """Class representing a stepwise decay.

    This class generates a stepwise decay function that maps input data `X`, `y` to a decreasing range from
    `max_value` to `min_value`.

    It is possible to specify one of `n_steps` or `step_size` to determine the behaviour of the decay.

    - If `step_size` is provided, the decay will be split into `n_samples // step_size` steps, each of which will
        decrease the value by `step_width = (max_value - min_value) / n_steps`.
    - If `n_steps` is provided, the decay will be split into `n_steps` steps, each of which will decrease the value
        by `step_width = (max_value - min_value) / n_steps`.

    Each *step* of length *step_size* has constant weight, and then decreases by `step_width` until the minimum value is
    reached.

    !!! warning
        It is up to the user to sort the dataset appropriately.

    Parameters
    ----------
    n_steps : int | None, default=None
        The total number of steps in the decay.
    step_size : int | None, default=None
        The number of samples for each step in the decay.
    min_value : float, default=0.
        The minimum value of the decay.
    max_value : float, default=1.
        The maximum value of the decay.

    Raises
    ------
    ValueError
        - If `min_value` is greater than `max_value`.
        - If no value or both values are provided for `n_steps` or `step_size`.
        - If `step_size` less than 0 or greater than the number of samples.
        - If `n_steps` less than 0 or greater than the number of samples.

    TypeError
        - If `n_steps` is not an integer.
        - If `step_size` is not an integer.
    """

    def __init__(self, n_steps=None, step_size=None, min_value=0.0, max_value=1.0):
        self.n_steps = n_steps
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, X, y):
        """Compute the decay values based on the shape of `y`.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features,)
            Training data. Unused, present for API consistency by convention.
        y : array-like, shape=(n_samples,)
            Target values. Used to determine the number of samples in the decay.

        Returns
        -------
        np.ndarray, shape=(n_samples,)
            The decay values.
        """
        if self.min_value > self.max_value:
            raise ValueError("`min_value` must be less than or equal to `max_value`")

        n_samples = y.shape[0]

        if self.step_size is None and self.n_steps is None:
            raise ValueError("Either `step_size` or `n_steps` must be provided")

        elif self.step_size is not None and self.n_steps is not None:
            raise ValueError("Only one of `step_size` or `n_steps` must be provided")

        elif self.step_size is not None and self.n_steps is None:
            if not isinstance(self.step_size, int):
                raise TypeError("`step_size` must be an integer")

            if self.step_size <= 0:
                raise ValueError("`step_size` must be greater than 0")

            if self.step_size > n_samples:
                raise ValueError(
                    "`step_size` must be less than or equal to the number of samples"
                )

            step_size = self.step_size
            n_steps = n_samples // self.step_size
            step_width = (self.max_value - self.min_value) / n_steps

        elif self.step_size is None and self.n_steps is not None:
            if not isinstance(self.n_steps, int):
                raise TypeError("`n_steps` must be an integer")

            if self.n_steps <= 0:
                raise ValueError("`n_steps` must be greater than 0")

            if self.n_steps > n_samples:
                raise ValueError(
                    "`n_steps` must be less than or equal to the number of samples"
                )

            step_size = n_samples // self.n_steps
            n_steps = self.n_steps
            step_width = (self.max_value - self.min_value) / n_steps

        return (
            self.max_value
            - (np.arange(n_samples, 0, -1) // self.step_size) * step_width
        )


class SigmoidDecay:
    """Class representing a sigmoid decay.

    This class generates a sigmoid decay function that maps input data `X`, `y` to a non-linearly decreasing range from
    `max_value` to `min_value`. The steepness of the decay is determined by the `growth_rate` parameter. If not provided
    this will be set to `10 / n_samples`, which is a good default for most cases.

    !!! warning
        It is up to the user to sort the dataset appropriately.

    Parameters
    ----------
    growth_rate : float | None, default=None
        The growth rate of the sigmoid function. If not provided this will be set to `10 / n_samples`.
    min_value : float, default=0.
        The minimum value of the decay.
    max_value : float, default=1.
        The maximum value of the decay.

    Raises
    ------
    ValueError
        - If `min_value` is greater than `max_value`.
        - If `growth_rate` is not between 0 and 1.
    """

    def __init__(self, growth_rate=None, min_value=0.0, max_value=1.0):
        self.growth_rate = growth_rate
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, X, y):
        """Compute the decay values based on the shape of `y`.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features,)
            Training data. Unused, present for API consistency by convention.
        y : array-like, shape=(n_samples,)
            Target values. Used to determine the number of samples in the decay.

        Returns
        -------
        np.ndarray, shape=(n_samples,)
            The decay values.
        """
        if self.min_value > self.max_value:
            raise ValueError("`min_value` must be less than or equal to `max_value`")

        if self.growth_rate is not None and (
            self.growth_rate <= 0 or self.growth_rate >= 1
        ):
            raise ValueError("`growth_rate` must be between 0. and 1.")

        n_samples = y.shape[0]
        growth_rate = self.growth_rate or 10 / n_samples

        return self.min_value + (self.max_value - self.min_value) * self._sigmoid(
            x=np.arange(n_samples), growth_rate=growth_rate, offset=n_samples // 2
        )

    @staticmethod
    def _sigmoid(x, growth_rate, offset):
        return 1 / (1 + np.exp(-growth_rate * (x - offset)))
