from functools import partial

import narwhals.stable.v1 as nw
import numpy as np
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list, expanding_list


def constant_shrinkage(group_sizes, alpha: float) -> np.ndarray:
    r"""The augmented prediction for each level is the weighted average between its prediction and the augmented
    prediction for its parent.

    Let $\hat{y}_i$ be the prediction at level $i$, with $i=0$ being the root, than the augmented prediction
    $\hat{y}_i^* = \alpha \hat{y}_i + (1 - \alpha) \hat{y}_{i-1}^*$, with $\hat{y}_0^* = \hat{y}_0$.

    Parameters
    ----------
    group_sizes : array-like
        The number of observations in each group, must implement the `__len__` method.
    alpha : float
        The weight of the prediction at the current level.

    Returns
    -------
    np.ndarray
        The weights for each group.
    """
    n_groups = len(group_sizes)
    return np.array(
        [alpha ** (n_groups - 1)]
        + [alpha ** (n_groups - 1 - i) * (1 - alpha) for i in range(1, n_groups - 1)]
        + [(1 - alpha)]
    )


def relative_shrinkage(group_sizes) -> np.ndarray:
    """Weigh each group according to its size.

    Parameters
    ----------
    group_sizes : array-like
        The number of observations in each group.

    Returns
    -------
    np.ndarray
        The weights for each group.
    """
    return np.asarray(group_sizes)


def no_shrinkage_function(x, n):
    # n = len(self.fitted_levels_[-1])
    return np.pad([1], (len(x) - 1, n - len(x)), "constant", constant_values=(0))


def min_n_obs_shrinkage(group_sizes, min_n_obs: int) -> np.ndarray:
    """Use only the smallest group with a certain amount of observations.

    Parameters
    ----------
    group_sizes : array-like
        The number of observations in each group.
    min_n_obs : int
        The minimum number of observations for a group to be considered.

    Returns
    -------
    np.ndarray
        The weights for each group.
    """
    if min_n_obs > max(group_sizes):
        raise ValueError(f"There is no group with size greater than or equal to {min_n_obs}")

    res = np.zeros(len(group_sizes))
    res[np.argmin(np.array(group_sizes) >= min_n_obs) - 1] = 1
    return res


def equal_shrinkage(group_sizes) -> np.ndarray:
    """Each group is weighed equally.

    Parameters
    ----------
    group_sizes : array-like
        The number of observations in each group, must implement the `__len__` method.

    Returns
    -------
    np.ndarray
        The weights for each group.
    """
    return np.ones(len(group_sizes))


class ShrinkageMixin:
    """Mixin class for shrinkage functionality (setting shrinkage, checking shrinkage function, and fitting shrinkage
    factors). The shrinkage factors are used to weigh the predictions of the different levels of a model.

    Class inherits from this mixin should have the following attributes:

    - `_ALLOWED_SHRINKAGE` : dict[str, callable]
        A dictionary mapping the name of the shrinkage function to the function itself.
    - `shrinkage` : str | callable | None
        The shrinkage function to use. If a callable is passed, it should take an array of group sizes and return an
        array of shrinkage factors.
        `shrinkage` is parsed by `_set_shrinkage_function`, which then returns `shrinkage_function_` to be used in
        `_fit_shrinkage_factors`.
    - `shrinkage_kwargs` : dict[str, Any]
        Additional keyword arguments to pass to the shrinkage function.
    - `fitted_levels_` : list[str | int]
        List of the levels that have been fitted.
    - `estimators_` : dict[tuple[Any, ...], scikit-learn compatible estimator]
    """

    def _set_shrinkage_function(self):
        """Set the shrinkage function and validate it if it is a custom callable"""
        if isinstance(self.shrinkage, str) and self.shrinkage in self._ALLOWED_SHRINKAGE.keys():
            shrinkage_function_ = self._ALLOWED_SHRINKAGE[self.shrinkage]

        elif callable(self.shrinkage):
            self.__check_shrinkage_func()
            shrinkage_function_ = self.shrinkage

        elif self.shrinkage is None:
            """Instead of keeping two different behaviors for shrinkage and non-shrinkage cases, this conditional block
            maps no shrinkage to a constant shrinkage function, with all the weight on the grouped passed,
            independently from the level sizes, as expected from the other shrinkage functions (*).
            This allows the rest of the code to be agnostic to the shrinkage function, and the shrinkage factors.

            (*) Consider the following example:

            - groups = ["a", "b"] with values (0, 0), (0, 1) and (1, 0) of respective sizes 6, 5, 9.
            - Considering these sizes, in `__fit_shrinkage_factors` the hierarchical_counts will be:
                - (1, 0, 0): [20, 11, 6]
                - (1, 0, 1): [20, 11, 5]
                - (1, 1, 0): [20, 9, 9]

                Notice that we always have the same total count (20), and the shrinkage factors will reflect that.
            - For `shrinkage = "relative"`, we get the following shrinkage factors:
                {
                    (1,): array([1.]),
                    (1, 0): array([0.64, 0.35]),
                    (1, 1): array([0.69, 0.31]),
                    (1, 0, 0): array([0.54, 0.30 , 0.16]),
                    (1, 0, 1): array([0.56, 0.30, 0.14]),
                    (1, 1, 0): array([0.52, 0.24, 0.24])
                }
            - For `shrinkage = None`, we get the following shrinkage factors:
                {
                    (1,): array([1., 0., 0.]),
                    (1, 0): array([0., 1., 0.]),
                    (1, 1): array([0., 1., 0.]),
                    (1, 0, 0): array([0., 0., 1.]),
                    (1, 0, 1): array([0., 0., 1.]),
                    (1, 1, 0): array([0., 0., 1.])
                }
            """

            shrinkage_function_ = partial(no_shrinkage_function, n=self.n_fitted_levels_)

        else:
            raise ValueError(
                f"`shrinkage` should be either `None`, {self._ALLOWED_SHRINKAGE.keys()}, or a callable. "
                f"Found {self.shrinkage} of type {type(self.shrinkage)}"
            )
        return shrinkage_function_

    def __check_shrinkage_func(self):
        """Validate the shrinkage function if a function is specified"""
        group_lengths = [10, 5, 2]
        expected_shape = np.asarray(group_lengths).shape
        try:
            result = self.shrinkage(group_lengths)
        except Exception as e:
            raise ValueError(f"Caught an exception while checking the shrinkage function: {str(e)}") from e
        else:
            if not isinstance(result, np.ndarray):
                raise ValueError(f"shrinkage_function({group_lengths}) should return an np.ndarray")
            if result.shape != expected_shape:
                raise ValueError(f"shrinkage_function({group_lengths}).shape should be {expected_shape}")

    def _fit_shrinkage_factors(self, frame, groups, most_granular_only=False):
        """Computes the shrinkage coefficients for fitted group values (corresponding to the keys of self.estimators_).

        Parameters
        ----------
        frame : pd.DataFrame
            The DataFrame to group by.
        groups : list[str | int]
            The columns to group by.
        most_granular_only : bool
            Whether to return only the shrinkage factors for the most granular group values.
        """
        check_is_fitted(self, ["estimators_", "shrinkage_function_"])
        counts = frame.group_by(groups).agg(nw.len().alias("counts"))
        all_grp_values = list(self.estimators_.keys())

        if most_granular_only:
            all_grp_values = [grp_value for grp_value in all_grp_values if len(as_list(grp_value)) == len(groups)]

        hierarchical_counts = {
            grp_value: [
                # As zip is "zip shortest" and filter works with comma separate conditions:
                counts.filter(*[nw.col(c) == v for c, v in zip(groups, subgroup)])
                .select(nw.sum("counts"))
                .to_numpy()[0][0]
                for subgroup in expanding_list(grp_value, tuple)
            ]
            for grp_value in all_grp_values
        }

        shrinkage_factors = {
            grp_value: self.shrinkage_function_(counts_, **(self.shrinkage_kwargs or {}))
            for grp_value, counts_ in hierarchical_counts.items()
        }

        # Normalize and pad
        return {grp_value: shrink_array / shrink_array.sum() for grp_value, shrink_array in shrinkage_factors.items()}
