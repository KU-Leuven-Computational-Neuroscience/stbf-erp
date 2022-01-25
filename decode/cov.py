from abc import ABC, abstractmethod

import numpy as np
from sklearn.covariance import EmpiricalCovariance, shrunk_covariance
import scipy.linalg
from sklearn.utils.validation import check_is_fitted


def oas(cov, n_epochs):
    """Calculate the Oracle Approximating Shrinkage coefficient

    Oracle Approximating Shrinkage for empirical covariance matrices [1] or
    Kronecker factor covariance matrices [2].

    [1] Y. Chen, A. Wiesel, Y. C. Eldar, and A. O. Hero, “Shrinkage algorithms
    for MMSE covariance estimation,” IEEE Transactions on Signal Processing,
    vol. 58, no. 10, Art. no. 10, Oct. 2010, doi: 10.1109/tsp.2010.2053029.

    [2] L. Xie, Z. He, J. Tong, T. Liu, J. Li, and J. Xi, “Regularized
    Estimation of Kronecker-Structured Covariance Matrix.” 2021.

    Parameters
    ----------
    cov : array-like of shape (n_features, n_features)
        The empirical covariance matrix before shrinkage.

    n_epochs : int
        The number of epochs.

    Returns
    -------
    shrinkage : float such that 0 <= shrinkage <= 1
        The shrinkage coefficient calculated by Oracle Approximating Shrinkage.
    """
    n_features = cov.shape[0]
    tr_cov = np.trace(cov)
    cov2 = cov.dot(cov)
    tr_cov2 = np.trace(cov2)

    num = tr_cov ** 2 + (1 - 2 / n_features) * tr_cov2
    den = (1 - n_epochs / n_features - (2 * n_epochs) / n_features ** 2) \
          * tr_cov ** 2 \
          + (n_epochs + 1 + (2 * (n_epochs - 1)) / n_features) * tr_cov2
    shrinkage = min(max(num / den, 0), 1)
    return shrinkage


def loocv(cov, n_epochs, mean_tr_covs_square):
    """Calculate the Leave-One-Out Cross-Validated shrinkage coefficient.

    Leave-One-Out Cross-Validated shrinkage coefficient for empirical
    covariance matrices [1] or Kronecker factor covariance matrices [2].

    [1] J. Tong, R. Hu, J. Xi, Z. Xiao, Q. Guo, and Y. Yu, “Linear shrinkage
    estimation of covariance matrices using low-complexity cross-validation,”
    Signal Processing, vol. 148, pp. 223–233, Jul. 2018, doi:
    10.1016/j.sigpro.2018.02.026.

    [2] L. Xie, Z. He, J. Tong, T. Liu, J. Li, and J. Xi,
    “Regularized Estimation of Kronecker-Structured Covariance Matrix.” 2021.

    Parameters
    ----------
    cov : ndarray of shape (n_features, n_features)
        The empirical covariance matrix before shrinkage.

    n_epochs : int
        The number of epochs.

    mean_tr_covs_square : float
        Can be calculated as (1/n_epochs)Sum(Trace(Xi.S^(-1).Xi^T)²) with S
        the previous covariance estimate.
        For non-iterative estimation
        procedures, S is the identity matrix and mean_tr_covs_square can thus
        be faster calculated as (1/n_epochs)Sum(||Xi||^4).

    Returns
    -------
    shrinkage : float such that 0 <= shrinkage <= 1
        The shrinkage coefficient calculated by Oracle Approximating Shrinkage.
    """
    p = cov.shape[0]
    tr_cov = np.trace(cov)
    cov_square = cov.dot(cov)
    tr_cov_square = np.trace(cov_square)
    target = (tr_cov / p) * np.eye(p, dtype=cov.dtype)
    target_square = target.dot(target)
    tr_target_square = np.trace(target_square)
    tr_cov_target = np.trace(cov.dot(target))

    num = (n_epochs * tr_cov_square) / (n_epochs - 1) \
          - 2 * tr_cov_target + tr_target_square \
          - mean_tr_covs_square / (n_epochs - 1)
    den = ((n_epochs ** 2 - 2 * n_epochs) * tr_cov_square) \
          / (n_epochs - 1) ** 2 \
          - 2 * tr_cov_target + tr_target_square \
          + mean_tr_covs_square / ((n_epochs - 1) ** 2)
    shrinkage = 1 - min(max(num / den, 0), 1)
    return shrinkage


class ToeplitzCovariance(EmpiricalCovariance):
    """Toeplitz structured covariance estimator

    A matrix is Toeplitz structured if it adheres to T_(i,j) = T_(i+1,j+1) .

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    def _set_covariance(self, covariance):
        # Force toeplitz
        toeplitz = self._force_toeplitz(covariance)
        covariance = scipy.linalg.toeplitz(toeplitz)
        # Set covariance
        self.covariance_ = covariance
        # Set precision
        if self.store_precision:
            self.precision_ = self._calc_precision()
        else:
            self.precision_ = None

    def _calc_precision(self):
        inv_cov = scipy.linalg.pinvh(self.covariance_)
        return inv_cov

    def _force_toeplitz(self, cov):
        """ Coerce the calculated empirical covariance to a Toeplitz-structured
        matrix by setting each diagonal to its mean value.
        """
        n_features = cov.shape[0]
        toeplitz = np.zeros_like(cov[:, 0])
        for i in range(n_features):
            d = np.diag(cov, k=i)
            toeplitz[i] = d.mean()
        return toeplitz

    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
        if self.store_precision:
            precision = self.precision_
        else:
            precision = self._calc_precision()
        return precision


class SpatiotemporalCovariance(EmpiricalCovariance, ABC):
    """Abstract base class for covariances of spatiotemporal epochs."""

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def right_dot(self, X):
        pass


class FullCovariance(SpatiotemporalCovariance):
    """Spatiotemporal covariance estimator based on flattened epochs.

    Calculate the spatiotemporal covariance as the empirical covariance matrix
    of the flattened epochs. Shrinkage can later be applied.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=True
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    shrinkage : 'oas', 'loocv', float such that 0<=shrinkage<=1 or None, default='loocv'
        'oas' will use Oracle Approximating Shrinkage, 'loocv' will use
        Leave-One-Out Cross-Validated shrinkage. None will apply no shrinkaage.

    Attributes
    ----------
    location_ : ndarray of shape (n_channels, n_samples)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_channels*n_samples, n_channels*n_samples)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_channels*n_samples, n_channels*n_samples)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)
    """

    def __init__(self, assume_centered=True, store_precision=True,
                 shrinkage='loocv'):
        super().__init__(assume_centered=assume_centered,
                         store_precision=store_precision)
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        """Fit the covariance estimator to the data.

        Parameters
        ----------
        X : array-like of shape (n_epochs, n_channels, n_samples)
         Spatiotemporal epochs as training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
           Returns the instance itself.
        """
        # Extract mean and center data
        if self.assume_centered:
            self.location_ = np.zeros_like(X[0])
        else:
            self.location_ = X.mean(0)
            X = X - np.mean(X, 0)

        # Flatten epochs
        X = X.reshape(X.shape[0], -1)
        covariance = np.cov(X.T)

        # Shrinkage
        self.shrinkage_ = self.shrinkage
        if self.shrinkage_ == 'loocv':
            mean_norm_x_4 = np.mean(scipy.linalg.norm(X, axis=1) ** 4)
            self.shrinkage_ = loocv(covariance, X.shape[0], mean_norm_x_4)
        elif self.shrinkage_ == 'oas':
            self.shrinkage_ = oas(covariance, X.shape[0])
        elif not self.shrinkage:
            self.shrinkage_ = 0
        else:
            self.shrinkage_ = self.shrinkage
        covariance = shrunk_covariance(covariance, self.shrinkage_)

        # Set covariance
        self._set_covariance(covariance)
        return self

    def right_dot(self, X):
        check_is_fitted(self)
        assert (len(X.shape) == 2)
        shape = X.shape
        transformed = self.get_precision().dot(X.flatten())
        return transformed.reshape(shape)

    def _set_covariance(self, covariance):
        # set covariance
        self.covariance_ = covariance
        # set precision
        if self.store_precision:
            self.precision_ = self._calc_precision()
        else:
            self.precision_ = None

    def get_precision(self):
        if self.store_precision:
            precision = self.precision_
        else:
            precision = self._calc_precision()
        return precision

    def _calc_precision(self):
        inv_cov = scipy.linalg.pinvh(self.covariance_)
        return inv_cov


class KroneckerCovariance(SpatiotemporalCovariance):
    """Estimator of a Kronecker product of spatial and temporal covariances.

    Fixed-Point Iteration algorithm [1] for the estimation of the
    spatiotemporal covariance as a Kronecker product of a spatial and temporal
    covariance matrix.
    Shrinkage can be applied to the spatial and temporal covariances at each
    iteration.

    [1] A. Wiesel, “On the convexity in Kronecker structured covariance
    estimation,” in 2012 IEEE Statistical Signal Processing Workshop (SSP),
    Aug. 2012, pp. 880–883. doi: 10.1109/SSP.2012.6319848.

    Parameters
    ----------
    assume_centered : bool, default=True
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    max_iter : int, default=128
        Maximum number of Fixed-Point Iterations.

    tol : float, default=1e-12
        Tolerance on the update of the norm of the covariance matrix to end
        the iteration procedure.

    verbose : bool, default=False
        Print covariance norm update at each iteration.

    shrinkage : tuple, default=('loocv','loocv')
         Shrinkage to apply to respectively to spatial and temporal covariance
         matrix estimates at each iteration. See `FullCovariance` for possible
         shrinkage values.

    spatial : EmpiricalCovariance, default=EmpiricalCovariance()
        Estimator for the spatial covariance

    temporal : EmpiricalCovariance, default=ToeplitzCovariance()
        Estimator for the temporal covariance


    Attributes
    ----------
    location_ : ndarray of shape (n_channels, n_samples)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_channels*n_samples, n_channels*n_samples)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_channels*n_samples, n_channels*n_samples)
        Estimated pseudo-inverse matrix.

    spatial_ : EmpiricalCovariance
        The fitted estimator of the spatial covariance.

    temporal_ : EmpiricalCovariance
        The fitted estimator of the spatial covariance

    iter_ : int
        The number of iterations.
    """

    def __init__(self, *, assume_centered=True, max_iter=128, tol=1e-12,
                 verbose=False, shrinkage=('loocv', 'loocv'), spatial=None,
                 temporal=None):
        super().__init__()
        self.assume_centered = assume_centered
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.shrinkage = shrinkage
        self.spatial = spatial
        self.temporal = temporal

    def fit(self, X, y=None):
        """Fit the covariance estimator to the data.

        Parameters
        ----------
        X : array-like of shape (n_epochs, n_channels, n_samples)
         Spatiotemporal epochs as training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
           Returns the instance itself.
        """
        # Extract mean and center data
        if self.assume_centered:
            self.location_ = np.zeros_like(X[0, :, :])
        else:
            self.location_ = X.mean(0)
            X = X - np.mean(X, axis=0)

        # Initialize spatial and temporal covariance estimators
        self.spatial_ = self.spatial
        self.temporal_ = self.temporal
        if self.spatial_ is None:
            self.spatial_ = EmpiricalCovariance()
        if self.temporal_ is None:
            self.temporal_ = ToeplitzCovariance()
        self.spatial_.set_params(assume_centered=True, store_precision=True)
        self.spatial_.set_params(assume_centered=True, store_precision=True)
        self._covariance_ = None
        self._precision_ = None

        # Execute Kronecker covariance estimation procedure
        self._fit_kronecker(X, y)

        return self

    def _fit_kronecker(self, X, y):
        n_epochs, n_channels, n_samples = X.shape
        X_conj = X.conj()
        XH = X_conj.transpose((0, 2, 1))

        # Initialize spatial and temporal covariance to identity matrix.
        self.iter_ = 0
        self.spatial_._set_covariance(np.eye(n_channels))
        self.temporal_._set_covariance(np.eye(n_samples))


        for self.iter_ in range(1, self.max_iter + 1):
            if self.verbose:
                print(f'Iteration {self.iter_}', end='  ')
            sp_cov = np.einsum('ijk,kl,ilm->jm', X, self.temporal_.precision_,
                               XH, optimize=['einsum_path', (0, 1), (0, 1)])
            tmp_cov = np.einsum('ijk,kl,ilm->jm', XH, self.spatial_.precision_,
                                X, optimize=['einsum_path', (0, 1), (0, 1)])
            sp_cov /= (n_epochs - 1)
            tmp_cov /= (n_epochs - 1)

            # Shrink
            ## Determine spatial shrinkage
            if self.shrinkage[0] == 'loocv':
                mean_tr_covs_2_sp = 0
                for i in range(n_epochs):
                    sp_cov_i = X[i].dot(self.temporal_.precision_).dot(XH[i])
                    mean_tr_covs_2_sp += np.trace(sp_cov_i) ** 2 / n_epochs
                sp_shrinkage = loocv(sp_cov, n_epochs, mean_tr_covs_2_sp)
            elif self.shrinkage[0] == 'oas':
                sp_shrinkage = oas(sp_cov, n_epochs)
            else:
                sp_shrinkage = self.shrinkage[0]
            ## Determine temporal shrinkage
            if self.shrinkage[1] == 'loocv':
                mean_tr_covs_2_tmp = 0
                for i in range(n_epochs):
                    tmp_cov_i = XH[i].dot(self.spatial_.precision_).dot(X[i])
                    mean_tr_covs_2_tmp += np.trace(tmp_cov_i) ** 2 / n_epochs
                tmp_shrinkage = loocv(tmp_cov, n_epochs, mean_tr_covs_2_tmp)
            elif self.shrinkage[1] == 'oas':
                tmp_shrinkage = oas(tmp_cov, n_epochs)
            else:
                tmp_shrinkage = self.shrinkage[1]
            ## Shrink covariances
            sp_cov = shrunk_covariance(sp_cov, sp_shrinkage)
            tmp_cov = shrunk_covariance(tmp_cov, tmp_shrinkage)

            # Normalize
            sp_cov /= np.trace(sp_cov) / n_channels
            tmp_cov /= np.trace(tmp_cov) / n_samples

            # Determine step
            step = scipy.linalg.norm(sp_cov - self.spatial_.covariance_) \
                   * scipy.linalg.norm(tmp_cov - self.temporal_.covariance_)

            # Set spatial covariance
            self.spatial_._set_covariance(sp_cov)
            # Set temporal covariance
            self.temporal_._set_covariance(tmp_cov)

            # Stop iteration if step<tol
            if self.verbose:
                print(f'step: {step}')
            if step < self.tol:
                break

    @property
    def covariance_(self):
        """Getter for the covariance matrix.

        Returns
        -------
        covariance_ : array-like of shape (n_channels*n_samples, n_channels*n_samples)
            The covariance matrix calculated as the Kronecker product of the spatial and
            temporal covariance matrices.
        """
        if self._covariance_ is None:
            self._covariance_ = np.kron(self.spatial_.covariance_,
                                        self.temporal_.covariance_)
        return self._covariance_

    @property
    def precision_(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like of shape (n_channels*n_samples, n_channels*n_samples)
            The precision matrix calculated as the Kronecker product of the spatial and
            temporal precision matrices.
        """
        if self._precision_ is None:
            self._precision_ = np.kron(self.spatial_.precision_,
                                       self.temporal_.precision_)
        return self._precision_

    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        covariance_ : array-like of shape (n_channels*n_samples, n_channels*n_samples)
            The precision matrix calculated as the Kronecker product of the spatial and
            temporal precision matrices.
        """
        return self.precision_

    def right_dot(self, X, y=None):
        """Efficienty multiply the precision matrix with an epoch.

        Parameters
        ----------
        X: array-like of shape (n_channels, n_samples)
            A spatiotemporal epoch to be transformed by the precision matrix.

        Returns
        -------
        transformed : array-like of shape (n_channel, n_samples)
            The transformed epoch precision.dot(X.T)
        """
        check_is_fitted(self)
        return self.spatial_.precision_.dot(X).dot(self.temporal_.precision_)


class NonIterativeKroneckerCovariance(KroneckerCovariance):
    """Estimator of a Kronecker product of spatial and temporal covariances.

    Efficient implementation of the Kronecker structured covariance estimator
    when  a single iteration of the Fixed-Point iteration algorithm is
    sufficient.

    Parameters
    ----------
    assume_centered : bool, default=True
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    verbose : bool, default=False
        Print covariance norm update at each iteration.

    shrinkage : tuple, default=('loocv','loocv')
         Shrinkage to apply to respectively to spatial and temporal covariance
         matrix estimates at each iteration. See `FullCovariance` for possible
         shrinkage values.

    spatial : EmpiricalCovariance, default=EmpiricalCovariance()
        Estimator for the spatial covariance

    temporal : EmpiricalCovariance, default=ToeplitzCovariance()
        Estimator for the temporal covariance


    Attributes
    ----------
    location_ : ndarray of shape (n_channels, n_samples)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_channels*n_samples, n_channels*n_samples)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_channels*n_samples, n_channels*n_samples)
        Estimated pseudo-inverse matrix.

    spatial_ : EmpiricalCovariance
        The fitted estimator of the spatial covariance.

    temporal_ : EmpiricalCovariance
        The fitted estimator of the spatial covariance
    """

    def __init__(self, *, assume_centered=True, shrinkage=('loocv', 'loocv'),
                 spatial=None, temporal=None):
        super().__init__(assume_centered=assume_centered, max_iter=1, tol=0,
                         verbose=False, shrinkage=shrinkage, spatial=spatial,
                         temporal=temporal)

    def _fit_kronecker(self, X, y):
        n_epochs, n_channels, n_samples = X.shape
        X_conj = np.conj(X)
        sp_cov = np.einsum('ijk,ilk->jl', X, X_conj, optimize=True)
        tmp_cov = np.einsum('ikj,ikl->jl', X, X_conj, optimize=True)
        sp_cov /= (n_epochs - 1)
        tmp_cov /= (n_epochs - 1)

        # Shrink
        self.spatial_shrinkage_ = self.shrinkage[0]
        if self.spatial_shrinkage_ == 'loocv':
            mean_norm_4 = np.mean(scipy.linalg.norm(X, axis=(1, 2)) ** 4)
            self.spatial_shrinkage_ = loocv(sp_cov, n_epochs, mean_norm_4)
        elif self.spatial_shrinkage_ == 'oas':
            self.spatial_shrinkage_ = oas(sp_cov, n_epochs)
        elif not self.spatial_shrinkage_:
            self.spatial_shrinkage_ = 0
        self.temporal_shrinkage_ = self.shrinkage[0]

        if self.temporal_shrinkage_ == 'loocv':
            mean_norm_4 = np.mean(scipy.linalg.norm(X, axis=(1, 2)) ** 4)
            self.temporal_shrinkage_ = loocv(tmp_cov, n_epochs, mean_norm_4)
        elif self.temporal_shrinkage_ == 'oas':
            self.temporal_shrinkage_ = oas(tmp_cov, n_epochs)
        elif not self.temporal_shrinkage_:
            self.temporal_shrinkage_ = 0

        sp_cov = shrunk_covariance(sp_cov, self.spatial_shrinkage_)
        tmp_cov = shrunk_covariance(tmp_cov, self.temporal_shrinkage_)

        # Normalize
        sp_cov /= np.trace(sp_cov) / n_channels
        tmp_cov /= np.trace(tmp_cov) / n_samples

        # Set and invert covariance
        self.spatial_._set_covariance(sp_cov)
        self.temporal_._set_covariance(tmp_cov)
