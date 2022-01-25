from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
from decode.cov import FullCovariance

class LCMVBeamformer(BaseEstimator, TransformerMixin):
    """Spatiotemporal LCMV beamformer

    Parameters
    ----------
    cov_estimator : SpatiotemporalCovariance, default=FullCovariance(shrinkage=False)
        Spatiotemporal covariance estimator

    lead_field: array-like of shape (n_channels, n_samples), default=None
        Lead field or activation pattern of the beamformer.
        If None, the lead field will be initialized to the difference of the
        average of target and the average of non-target epochs.

    Attributes
    ----------
    lead_field_ : array-like of shape (n_channels, n_samples)
        The lead field or activation pattern of the beamformer.

    weights_: array-like of shape (n_channels, n_samples)
        The LCMV-beamformer filter weights
    """

    def __init__(self, cov_estimator=None, lead_field=None):
        self.cov_estimator = cov_estimator
        self.lead_field = lead_field

    def fit(self, X, y=None):
        """Fit the beamformer to the data.

        Parameters
        ----------
        X : array-like of shape (n_epochs, n_channels, n_samples)
         Spatiotemporal epochs as training data.

        y : None or array-like of shape (n_epochs,)
            If y is not None and lead_field is None, y will be used to initialize
            the lead field.
            If y is None, a custom lead field needs to be specified.

        Returns
        -------
        self : object
           Returns the instance itself.
        """
        y = y.astype(bool)
        # Calculate activation pattern
        self.lead_field_ = self.lead_field
        if self.lead_field_ is None:
            if y is not None:
                avg_target = np.mean(X[y, :], axis=0)
                avg_non_target = np.mean(X[~y, :], axis=0)
                self.lead_field_ = avg_target - avg_non_target
            else:
                self.lead_field_ = np.mean(X, axis=0)

        # Calculate covariance and precision
        self.cov_estimator_ = self.cov_estimator
        if self.cov_estimator_ is None:
            self.cov_estimator_ = FullCovariance(shrinkage=False)
        self.cov_estimator_.fit(X)

        # Calculate weights
        self.weights_ = self.cov_estimator_.right_dot(self.lead_field_)
        self.weights_ /= np.sum(self.lead_field_.conj() * self.weights_)
        return self

    def transform(self, X=None, y=None):
        """Transform epochs with the fitted beamformer.

        Parameters
        ----------
        X : array-like of shape (n_epochs, n_channels, n_samples)
         Epochs to be transformed by the beamformer filter.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        score : array-like of shape (n_epochs,1)
           Scalar score per epoch indicating to what extent the signal
           specified by the lead field is present in that epoch.
        """
        score = self.decision_function(X)
        return score[:, np.newaxis]

    def decision_function(self, X):
        """Score each epoch based on the presence of the lead field signal.

        Parameters
        ----------
        X : array-like of shape (n_epochs, n_channels, n_samples)
         Epochs to be transformed by the beamformer filter.

        Returns
        -------
        score : array-like of shape (n_epochs,1)
           Scalar score per epoch indicating to what extent the signal
           specified by the lead field is present in that epoch.
        """
        check_is_fitted(self)
        X = X.reshape(X.shape[0], -1)
        w = self.weights_.flatten()
        score = X.dot(w.T.conj())
        return score
