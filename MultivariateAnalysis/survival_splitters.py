from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._split import _RepeatedSplits, GroupsConsumerMixin
from sklearn.utils.validation import check_array
import numpy as np
from typing import Callable, Tuple, List, Optional

def event_observed_strata(y):   
    if isinstance(y, dict):
        time = np.asarray(y["time"])
        event = np.asarray(y["event"])
    else:
        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] >= 2:
            event, time = y_arr[:, 0], y_arr[:, 1]
        elif y_arr.ndim == 1:
            names = y_arr.dtype.names
            if "time" not in names or "event" not in names:
                raise ValueError("y must be (n,2) [event,time], (n,) with dtypes ['time','event'] or dict {'time','event'}.")
            else:
                time = y["time"]
                event = y["event"]
        else:
            raise ValueError("y must be (n,2) [event,time], (n,) with dtypes ['time','event'] or dict {'time','event'}.")
    return event


class SurvivalStratifyMixin:
    """Modify how y is read by splitters

    Applied a function to y in an sklearn splitter. By default is used to
    make survival data in form [time, event] usable by stratified splitters
    """
    def __init__(self, *args, y_transformer: Callable = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_transformer = y_transformer or event_observed_strata

    def split(self, X, y=None, groups=None):
        y_strata = None if y is None else self.y_transformer(y)
        # Delegate to the stratified base splitter
        for tr, te in super().split(X, y=y_strata, groups=groups):
            yield tr, te



def _compress_group_y(
    y: Optional[np.ndarray],
    groups: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
    """
    Check y is constant within each group and return:
      - group_ids: unique group labels (len G)
      - members: list of index arrays, one per group
      - group_y: y reduced to one value per group (None if y is None)
    """
    group_ids, group_inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(group_inv == gi) for gi in range(len(group_ids))]

    if y is None:
        return group_ids, members, None

    y = np.asarray(y)
    if y.shape[0] != groups.shape[0]:
        raise ValueError("y and groups must have the same length.")

    group_y = []
    for gi, idx in enumerate(members):
        ref = y[idx[0]]
        # Assert constant within group
        if not all(y[i] == ref for i in idx):
            raise ValueError(
                f"Strict group check failed: y differs within group {group_ids[gi]} "
            )
        group_y.append(ref)

    return group_ids, members, np.asarray(group_y)


class StrictGroupMixin:
    """Split data at group level

    Modifies an sklearn splitter to split a dataset at the group level rather than
    the sample level. Requires all elements in a group to have the same y variable.
    """
    def _compress(self, X, y, groups):
        if groups is None:
            raise ValueError("`groups` must be provided for StrictGroupSplitMixin.")
        group_ids, members, group_y = _compress_group_y(
            y, groups
        )
        # Placeholder X data
        Xg = np.empty((len(group_ids), 0))
        return Xg, group_y, group_ids, members
    
    def split(self, X, y=None, groups=None):
        Xg, yg, group_ids, members = self._compress(X, y, groups)

        # Delegate to base splitter at GROUP level
        for g_train, g_test in super().split(Xg, y=yg, groups=group_ids):
            # Map group indices back to sample indices
            if len(g_train):
                train_idx = np.concatenate([members[i] for i in g_train]).astype(int, copy=False)
            else:
                train_idx = np.empty(0, dtype=int)

            if len(g_test):
                test_idx = np.concatenate([members[i] for i in g_test]).astype(int, copy=False)
            else:
                test_idx = np.empty(0, dtype=int)

            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        # Compute n_splits at the group level (mirrors split behavior)
        Xg, yg, group_ids, _ = self._compress(X, y, groups)
        return super().get_n_splits(X=Xg, y=yg, groups=group_ids)



class SurvivalStratifiedGroupKFold(StratifiedGroupKFold):
    def __init__(self, n_splits = 5, shuffle = False, random_state = None, event_col = "event"):
        super().__init__(n_splits, shuffle, random_state)
        self._event_col = event_col

    def split(self, X, y = None, groups = None):
        for train, test in super().split(X, y[self._event_col], groups):
            yield train, test

class SurvivalRepeatedStratifiedGroupKFold(SurvivalStratifiedGroupKFold):
    def __init__(self, n_splits = 5, shuffle = False, n_repeats = 10, random_state = None, event_col = "event"):
        super().__init__(n_splits, shuffle, random_state)
        self._event_col = event_col
        self._n_repeats = n_repeats

    def split(self, X, y = None, groups = None):
        for _ in range(self._n_repeats):
            for train, test in super().split(X, y, groups):
                yield train, test

    def get_n_splits(self, X = None, y = None, groups = None):
        return super().get_n_splits(X, y, groups)*self._n_repeats
    
class RepeatedStratifiedGroupKFold(GroupsConsumerMixin, _RepeatedSplits):
    def __init__(self, *, n_splits=5, n_repeats=1, random_state=None):
        super().__init__(
            StratifiedGroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits
        )

    def split(self, X, y, groups=None):
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups=groups)