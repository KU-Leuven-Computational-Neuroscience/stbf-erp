import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class UnitVarianceChannelScaler(BaseEstimator, TransformerMixin):
    """Scale epochs to unit channel variance.

    Scale an epoched signal to unit channel variance by dividing each channel
    by its standard deviation.

    Attributes
    ----------
    std_: ndarray of shape (n_channels,)
        Channel standard deviations.
    """

    def fit(self, X, y=None):
        self.std_ = np.std(X, axis=(0, 2))
        return self

    def transform(self, X, y=None):
        return X / self.std_[np.newaxis, :, np.newaxis]
