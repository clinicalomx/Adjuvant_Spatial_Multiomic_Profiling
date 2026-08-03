import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted



class UniqueValueFilter(SelectorMixin, BaseEstimator):
    """_summary_

    Inspired by LowInfoFilter from STABL
    https://doi.org/10.1038/s41587-023-02033-x

    Args:
        SelectorMixin (_type_): _description_
        BaseEstimator (_type_): _description_
    """
    def __init__(self, unique_fraction=0.8):
        self.unique_fraction = unique_fraction
        self.n_samples = None
        self.unique_counts = None
        self.contains_nan = None

    def fit(self, X, y=None):

        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            force_all_finite="allow-nan",
        )

        if self.unique_fraction > 1 or self.unique_fraction < 0:
            raise ValueError(
                f"Unique fraction should be between 0 and 1 Got: {self.unique_fraction}"
            )
        
        n_samples = X.shape[0]
        self.n_samples = n_samples
        self.contains_nan = nan_columns = np.isnan(X).any(axis=0)

        # Account for nan being a unique value in numpy
        unique_counts = np.apply_along_axis(lambda x: (np.unique(x)).shape[0], 0, X)
        self.unique_counts = unique_counts - self.contains_nan
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self)

        return self.unique_counts >= self.unique_fraction * self.n_samples
    
    def _more_tags(self):
        return {"allow_nan": True}