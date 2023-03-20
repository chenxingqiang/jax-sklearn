
import numpy as np

def log1pexp(x):
    if x <= -37:
        return np.exp(x)
    elif x <= -2:
        return np.log1p(np.exp(x))
    elif x <= 18:
        return np.log(1. + np.exp(x))
    elif x <= 33.3:
        return x + np.exp(-x)
    else:
        return x

def sum_exp_minus_max(i, raw_prediction):
    n_classes = raw_prediction.shape[1]
    max_value = raw_prediction[i, 0]
    sum_exps = 0

    # Compute max value of array for numerical stability
    for k in range(1, n_classes):
        if max_value < raw_prediction[i, k]:
            max_value = raw_prediction[i, k]

    p = np.zeros(n_classes + 2, dtype=np.float64)
    for k in range(n_classes):
        p[k] = np.exp(raw_prediction[i, k] - max_value)
        sum_exps += p[k]

    p[n_classes] = max_value     # same as p[-2]
    p[n_classes + 1] = sum_exps  # same as p[-1]

    return p


class GradHessPair:
    def __init__(self, gradient, hessian):
        self.gradient = gradient
        self.hessian = hessian

# Half Squared Error
def loss_half_squared_error(y_true, raw_prediction):
    return 0.5 * (raw_prediction - y_true) * (raw_prediction - y_true)

def gradient_half_squared_error(y_true, raw_prediction):
    return raw_prediction - y_true

def grad_hess_half_squared_error(y_true, raw_prediction):
    return GradHessPair(raw_prediction - y_true, 1.0)

# Absolute Error
def loss_absolute_error(y_true, raw_prediction):
    return np.abs(raw_prediction - y_true)

def gradient_absolute_error(y_true, raw_prediction):
    return 1.0 if raw_prediction > y_true else -1.0

def grad_hess_absolute_error(y_true, raw_prediction):
    return GradHessPair(1.0 if raw_prediction > y_true else -1.0, 1.0)

# Quantile Loss / Pinball Loss
def loss_pinball_loss(y_true, raw_prediction, quantile):
    return (quantile * (y_true - raw_prediction) if y_true >= raw_prediction
            else (1.0 - quantile) * (raw_prediction - y_true))

def gradient_pinball_loss(y_true, raw_prediction, quantile):
    return -quantile if y_true >= raw_prediction else 1.0 - quantile

def grad_hess_pinball_loss(y_true, raw_prediction, quantile):
    return GradHessPair(-quantile if y_true >= raw_prediction else 1.0 - quantile, 1.0)

# Half Poisson Deviance with Log-Link, dropping constant terms
def loss_half_poisson(y_true, raw_prediction):
    return np.exp(raw_prediction) - y_true * raw_prediction

def gradient_half_poisson(y_true, raw_prediction):
    return np.exp(raw_prediction) - y_true

def grad_hess_half_poisson(y_true, raw_prediction):
    return GradHessPair(np.exp(raw_prediction) - y_true, np.exp(raw_prediction))

# Half Gamma Deviance with Log-Link, dropping constant terms
def loss_half_gamma(y_true, raw_prediction):
    return raw_prediction + y_true * np.exp(-raw_prediction)

def gradient_half_gamma(y_true, raw_prediction):
    return 1.0 - y_true * np.exp(-raw_prediction)

def grad_hess_half_gamma(y_true, raw_prediction):
    exp_neg_raw_prediction = np.exp(-raw_prediction)
    return GradHessPair(1.0 - y_true * exp_neg_raw_prediction, y_true * exp_neg_raw_prediction)


# Half Tweedie Deviance with Log-Link, dropping constant terms
# Note that by dropping constants this is no longer continuous in parameter power.
def loss_half_tweedie(y_true, raw_prediction, power):
    if power == 0.0:
        return loss_half_squared_error(y_true, np.exp(raw_prediction))
    elif power == 1.0:
        return loss_half_poisson(y_true, raw_prediction)
    elif power == 2.0:
        return loss_half_gamma(y_true, raw_prediction)
    else:
        return (np.exp((2.0 - power) * raw_prediction) / (2.0 - power)
                - y_true * np.exp((1.0 - power) * raw_prediction) / (1.0 - power))

def gradient_half_tweedie(y_true, raw_prediction, power):
    if power == 0.0:
        exp1 = np.exp(raw_prediction)
        return exp1 * (exp1 - y_true)
    elif power == 1.0:
        return gradient_half_poisson(y_true, raw_prediction)
    elif power == 2.0:
        return gradient_half_gamma(y_true, raw_prediction)
    else:
        return (np.exp((2.0 - power) * raw_prediction)
                - y_true * np.exp((1.0 - power) * raw_prediction))

def loss_grad_half_tweedie(y_true, raw_prediction, power):
    if power == 0.0:
        exp1 = np.exp(raw_prediction)
        loss = loss_half_squared_error(y_true, exp1)
        gradient = exp1 * (exp1 - y_true)
    elif power == 1.0:
        loss, gradient = loss_half_poisson(y_true, raw_prediction)
    elif power == 2.0:
        loss, gradient = loss_half_gamma(y_true, raw_prediction)
    else:
        exp1 = np.exp((1.0 - power) * raw_prediction)
        exp2 = np.exp((2.0 - power) * raw_prediction)
        loss = exp2 / (2.0 - power) - y_true * exp1 / (1.0 - power)
        gradient = exp2 - y_true * exp1

    return GradHessPair(loss, gradient)

def grad_hess_half_tweedie(y_true, raw_prediction, power):
    if power == 0.0:
        exp1 = np.exp(raw_prediction)
        gradient = exp1 * (exp1 - y_true)
        hessian = exp1 * (2 * exp1 - y_true)
    elif power == 1.0:
        gradient, hessian = gradient_half_poisson(y_true, raw_prediction)
    elif power == 2.0:
        gradient, hessian = gradient_half_gamma(y_true, raw_prediction)
    else:
        exp1 = np.exp((1.0 - power) * raw_prediction)
        exp2 = np.exp((2.0 - power) * raw_prediction)
        gradient = exp2 - y_true * exp1
        hessian = (2.0 - power) * exp2 - (1.0 - power) * y_true * exp1

    return GradHessPair(gradient, hessian)



# Half Tweedie Deviance with identity link, without dropping constant terms!
# Therefore, best loss value is zero.
def loss_half_tweedie_identity(y_true, raw_prediction, power):
    if power == 0.0:
        return loss_half_squared_error(y_true, raw_prediction)
    elif power == 1.0:
        if y_true == 0:
            return raw_prediction
        else:
            return y_true * np.log(y_true / raw_prediction) + raw_prediction - y_true
    elif power == 2.0:
        return np.log(raw_prediction / y_true) + y_true / raw_prediction - 1.0
    else:
        tmp = pow(raw_prediction, 1.0 - power)
        tmp = raw_prediction * tmp / (2.0 - power) - y_true * tmp / (1.0 - power)
        if y_true > 0:
            tmp += pow(y_true, 2.0 - power) / ((1.0 - power) * (2.0 - power))
        return tmp

def gradient_half_tweedie_identity(y_true, raw_prediction, power):
    if power == 0.0:
        return raw_prediction - y_true
    elif power == 1.0:
        return 1.0 - y_true / raw_prediction
    elif power == 2.0:
        return (raw_prediction - y_true) / (raw_prediction * raw_prediction)
    else:
        return pow(raw_prediction, -power) * (raw_prediction - y_true)

def loss_grad_half_tweedie_identity(y_true, raw_prediction, power):
    if power == 0.0:
        gradient = raw_prediction - y_true
        loss = 0.5 * gradient * gradient
    elif power == 1.0:
        if y_true == 0:
            loss = raw_prediction
        else:
            loss = y_true * np.log(y_true / raw_prediction) + raw_prediction - y_true
        gradient = 1.0 - y_true / raw_prediction
    elif power == 2.0:
        loss = np.log(raw_prediction / y_true) + y_true / raw_prediction - 1.0
        tmp = raw_prediction * raw_prediction
        gradient = (raw_prediction - y_true) / tmp
    else:
        tmp = pow(raw_prediction, 1.0 - power)
        loss = raw_prediction * tmp / (2.0 - power) - y_true * tmp / (1.0 - power)
        if y_true > 0:
            loss += pow(y_true, 2.0 - power) / ((1.0 - power) * (2.0 - power))
        gradient = tmp * (1.0 - y_true / raw_prediction)

    return GradHessPair(loss, gradient)

def grad_hess_half_tweedie_identity(y_true, raw_prediction, power):
    if power == 0.0:
        gradient = raw_prediction - y_true
        hessian = 1.0
    elif power == 1.0:
        gradient = 1.0 - y_true / raw_prediction
        hessian = y_true / (raw_prediction * raw_prediction)
    elif power == 2.0:
        tmp = raw_prediction * raw_prediction
        gradient = (raw_prediction - y_true) / tmp
        hessian = (-1.0 + 2.0 * y_true / raw_prediction) / tmp
    else:
        tmp = pow(raw_prediction, -power)
        gradient = tmp * (raw_prediction - y_true)
        hessian = tmp * ((1.0 - power) + power * y_true / raw_prediction)



# Half Binomial deviance with logit-link, aka log-loss or binary cross entropy

import numpy as np

def loss_half_binomial(y_true, raw_prediction):
    return np.log1p(np.exp(raw_prediction)) - y_true * raw_prediction

def gradient_half_binomial(y_true, raw_prediction):
    exp_tmp = np.exp(-raw_prediction)
    return ((1 - y_true) - y_true * exp_tmp) / (1 + exp_tmp)

class GradHessPair:
    def __init__(self, gradient, hessian):
        self.gradient = gradient
        self.hessian = hessian

def loss_grad_half_binomial(y_true, raw_prediction):
    if raw_prediction <= 0:
        exp_tmp = np.exp(raw_prediction)
        if raw_prediction <= -37:
            loss = exp_tmp - y_true * raw_prediction
        else:
            loss = np.log1p(exp_tmp) - y_true * raw_prediction
        gradient = ((1 - y_true) * exp_tmp - y_true) / (1 + exp_tmp)
    else:
        exp_tmp = np.exp(-raw_prediction)
        if raw_prediction <= 18:
            loss = np.log1p(exp_tmp) + (1 - y_true) * raw_prediction
        else:
            loss = exp_tmp + (1 - y_true) * raw_prediction
        gradient = ((1 - y_true) - y_true * exp_tmp) / (1 + exp_tmp)

    return GradHessPair(gradient, loss)

def grad_hess_half_binomial(y_true, raw_prediction):
    exp_tmp = np.exp(-raw_prediction)
    gradient = ((1 - y_true) - y_true * exp_tmp) / (1 + exp_tmp)
    hessian = exp_tmp / (1 + exp_tmp)**2
    return GradHessPair(gradient, hessian)


# ---------------------------------------------------
# Extension Types for Loss Functions of 1-dim targets
# ---------------------------------------------------
class LossFunction:
    """Base class for convex loss functions."""

    def loss(self, y_true, raw_prediction):
        """Compute the loss for a single sample.

        Parameters
        ----------
        y_true : float
            Observed, true target value.
        raw_prediction : float
            Raw prediction value (in link space).

        Returns
        -------
        float
            The loss evaluated at `y_true` and `raw_prediction`.
        """
        pass

    def gradient(self, y_true, raw_prediction):
        """Compute gradient of loss w.r.t. raw_prediction for a single sample.

        Parameters
        ----------
        y_true : float
            Observed, true target value.
        raw_prediction : float
            Raw prediction value (in link space).

        Returns
        -------
        float
            The derivative of the loss function w.r.t. `raw_prediction`.
        """
        pass

    def grad_hess(self, y_true, raw_prediction):
        """Compute gradient and hessian.

        Gradient and hessian of loss w.r.t. raw_prediction for a single sample.

        This is usually diagonal in raw_prediction_i and raw_prediction_j.
        Therefore, we return the diagonal element i=j.

        For a loss with a non-canonical link, this might implement the diagonal
        of the Fisher matrix (=expected hessian) instead of the hessian.

        Parameters
        ----------
        y_true : float
            Observed, true target value.
        raw_prediction : float
            Raw prediction value (in link space).

        Returns
        -------
        tuple
            Gradient and hessian of the loss function w.r.t. `raw_prediction`.
        """
        pass


class LossFunction:
    # ... (previous methods defined earlier)

    def loss(self, y_true, raw_prediction, sample_weight, loss_out, n_threads=1):
        """Compute the pointwise loss value for each input."""
        pass

    def gradient(self, y_true, raw_prediction, sample_weight, gradient_out, n_threads=1):
        """Compute gradient of loss w.r.t raw_prediction for each input."""
        pass

    def loss_gradient(self, y_true, raw_prediction, sample_weight, loss_out, gradient_out, n_threads=1):
        """Compute loss and gradient of loss w.r.t raw_prediction."""
        self.loss(y_true, raw_prediction, sample_weight, loss_out, n_threads)
        self.gradient(y_true, raw_prediction, sample_weight, gradient_out, n_threads)
        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient_hessian(self, y_true, raw_prediction, sample_weight, gradient_out, hessian_out, n_threads=1):
        """Compute gradient and hessian of loss w.r.t raw_prediction."""
        pass


import numpy as np

class CyHalfSquaredError:
    """Half Squared Error with identity link.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction
    """

    def cy_loss(self, y_true, raw_prediction):
        return closs_half_squared_error(y_true, raw_prediction)

    def cy_gradient(self, y_true, raw_prediction):
        return cgradient_half_squared_error(y_true, raw_prediction)

    def cy_grad_hess(self, y_true, raw_prediction):
        return cgrad_hess_half_squared_error(y_true, raw_prediction)

    def loss(self, y_true, raw_prediction, sample_weight, loss_out, n_threads=1):
        n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in range(n_samples):
                loss_out[i] = closs_half_squared_error(y_true[i], raw_prediction[i])
        else:
            for i in range(n_samples):
                loss_out[i] = sample_weight[i] * closs_half_squared_error(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)


    def gradient(self, y_true, raw_prediction, sample_weight, gradient_out, n_threads=1):
        n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in range(n_samples):
                gradient_out[i] = cgradient_half_squared_error(y_true[i], raw_prediction[i])
        else:
            for i in range(n_samples):
                gradient_out[i] = sample_weight[i] * cgradient_half_squared_error(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(self, y_true, raw_prediction, sample_weight, gradient_out, hessian_out, n_threads=1):
        n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in range(n_samples):
                dbl2 = cgrad_hess_half_squared_error(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in range(n_samples):
                dbl2 = cgrad_hess_half_squared_error(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)


import numpy as np
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

class HalfSquaredError:
    """Half Squared Error with identity link.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction
    """

    @staticmethod
    def loss(y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = len(y_true)
        loss_out = np.empty(n_samples)

        if sample_weight is None:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i, loss_val in enumerate(executor.map(lambda yt, rp: 0.5 * (yt - rp) ** 2, y_true, raw_prediction)):
                    loss_out[i] = loss_val
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i, loss_val in enumerate(executor.map(lambda yt, rp, sw: sw * 0.5 * (yt - rp) ** 2, y_true, raw_prediction, sample_weight)):
                    loss_out[i] = loss_val

        return loss_out

    @staticmethod
    def gradient(y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = len(y_true)
        gradient_out = np.empty(n_samples)

        if sample_weight is None:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i, grad_val in enumerate(executor.map(lambda yt, rp: -(yt - rp), y_true, raw_prediction)):
                    gradient_out[i] = grad_val
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i, grad_val in enumerate(executor.map(lambda yt, rp, sw: sw * -(yt - rp), y_true, raw_prediction, sample_weight)):
                    gradient_out[i] = grad_val

        return gradient_out

    @staticmethod
    def gradient_hessian(y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = len(y_true)
        gradient_out = np.empty(n_samples)
        hessian_out = np.empty(n_samples)

        if sample_weight is None:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i, (grad_val, hess_val) in enumerate(executor.map(lambda yt, rp: (-(yt - rp), 1), y_true, raw_prediction)):
                    gradient_out[i] = grad_val
                    hessian_out[i] = hess_val
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for i, (grad_val, hess_val) in enumerate(executor.map(lambda yt, rp, sw: (sw * -(yt - rp), sw), y_true, raw_prediction, sample_weight)):
                    gradient_out[i] = grad_val
                    hessian_out[i] = hess_val

        return gradient_out, hessian_out


cdef class CyPinballLoss(CyLossFunction):
    """Quantile Loss aka Pinball Loss with identity link.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    Note: 2 * cPinballLoss(quantile=0.5) equals cAbsoluteError()
    """

    def __init__(self, quantile):
        self.quantile = quantile

    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_pinball_loss(y_true, raw_prediction, self.quantile)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_pinball_loss(y_true, raw_prediction, self.quantile)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_pinball_loss(y_true, raw_prediction, self.quantile)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_pinball_loss(y_true[i], raw_prediction[i], self.quantile)

        return np.asarray(loss_out)


    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_pinball_loss(y_true[i], raw_prediction[i], self.quantile)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_pinball_loss(y_true[i], raw_prediction[i], self.quantile)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfPoissonLoss(CyLossFunction):
    """Half Poisson deviance loss with log-link.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Poisson deviance with log-link is
        y_true * log(y_true/y_pred) + y_pred - y_true
        = y_true * log(y_true) - y_true * raw_prediction
          + exp(raw_prediction) - y_true

    Dropping constant terms, this gives:
        exp(raw_prediction) - y_true * raw_prediction
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_poisson(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_poisson(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_poisson(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_poisson(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_poisson(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_poisson(y_true[i], raw_prediction[i])
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_poisson(y_true[i], raw_prediction[i])
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_poisson(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_poisson(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_poisson(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_poisson(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfGammaLoss(CyLossFunction):
    """Half Gamma deviance loss with log-link.

    Domain:
    y_true and y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Gamma deviance with log-link is
        log(y_pred/y_true) + y_true/y_pred - 1
        = raw_prediction - log(y_true) + y_true * exp(-raw_prediction) - 1

    Dropping constant terms, this gives:
        raw_prediction + y_true * exp(-raw_prediction)
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_gamma(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_gamma(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_gamma(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_gamma(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_gamma(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_gamma(y_true[i], raw_prediction[i])
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_gamma(y_true[i], raw_prediction[i])
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_gamma(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_gamma(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_gamma(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_gamma(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfTweedieLoss(CyLossFunction):
    """Half Tweedie deviance loss with log-link.

    Domain:
    y_true in real numbers if p <= 0
    y_true in non-negative real numbers if 0 < p < 2
    y_true in positive real numbers if p >= 2
    y_pred and power in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Tweedie deviance with log-link and p=power is
        max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * y_pred**(1-p) / (1-p)
        + y_pred**(2-p) / (2-p)
        = max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * exp((1-p) * raw_prediction) / (1-p)
        + exp((2-p) * raw_prediction) / (2-p)

    Dropping constant terms, this gives:
        exp((2-p) * raw_prediction) / (2-p)
        - y_true * exp((1-p) * raw_prediction) / (1-p)

    Notes:
    - Poisson with p=1 and and Gamma with p=2 have different terms dropped such
      that cHalfTweedieLoss is not continuous in p=power at p=1 and p=2.
    - While the Tweedie distribution only exists for p<=0 or p>=1, the range
      0<p<1 still gives a strictly consistent scoring function for the
      expectation.
    """

    def __init__(self, power):
        self.power = power

    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_tweedie(y_true, raw_prediction, self.power)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_tweedie(y_true, raw_prediction, self.power)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_tweedie(y_true, raw_prediction, self.power)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_tweedie(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_tweedie(y_true[i], raw_prediction[i], self.power)

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_tweedie(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_tweedie(y_true[i], raw_prediction[i], self.power)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfTweedieLossIdentity(CyLossFunction):
    """Half Tweedie deviance loss with identity link.

    Domain:
    y_true in real numbers if p <= 0
    y_true in non-negative real numbers if 0 < p < 2
    y_true in positive real numbers if p >= 2
    y_pred and power in positive real numbers, y_pred may be negative for p=0.

    Link:
    y_pred = raw_prediction

    Half Tweedie deviance with identity link and p=power is
        max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * y_pred**(1-p) / (1-p)
        + y_pred**(2-p) / (2-p)

    Notes:
    - Here, we do not drop constant terms in contrast to the version with log-link.
    """

    def __init__(self, power):
        self.power = power

    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_tweedie_identity(y_true, raw_prediction, self.power)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_tweedie_identity(y_true, raw_prediction, self.power)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_tweedie_identity(y_true, raw_prediction, self.power)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_tweedie_identity(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)

cdef class CyHalfBinomialLoss(CyLossFunction):
    """Half Binomial deviance loss with logit link.

    Domain:
    y_true in [0, 1]
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)
    """


    cdef inline double cy_loss(self, double y_true, double raw_prediction) nogil:
        return closs_half_binomial(y_true, raw_prediction)

    cdef inline double cy_gradient(self, double y_true, double raw_prediction) nogil:
        return cgradient_half_binomial(y_true, raw_prediction)

    cdef inline double_pair cy_grad_hess(self, double y_true, double raw_prediction) nogil:
        return cgrad_hess_half_binomial(y_true, raw_prediction)

    def loss(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = closs_half_binomial(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                loss_out[i] = sample_weight[i] * closs_half_binomial(y_true[i], raw_prediction[i])

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] loss_out,        # OUT
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_binomial(y_true[i], raw_prediction[i])
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = closs_grad_half_binomial(y_true[i], raw_prediction[i])
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = cgradient_half_binomial(y_true[i], raw_prediction[i])
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                gradient_out[i] = sample_weight[i] * cgradient_half_binomial(y_true[i], raw_prediction[i])

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,          # IN
        Y_DTYPE_C[::1] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,   # IN
        G_DTYPE_C[::1] gradient_out,    # OUT
        G_DTYPE_C[::1] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i
            int n_samples = y_true.shape[0]
            double_pair dbl2

        if sample_weight is None:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_binomial(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in prange(
                n_samples, schedule='static', nogil=True, num_threads=n_threads
            ):
                dbl2 = cgrad_hess_half_binomial(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return np.asarray(gradient_out), np.asarray(hessian_out)


# The multinomial deviance loss is also known as categorical cross-entropy or
# multinomial log-likelihood
cdef class CyHalfMultinomialLoss(CyLossFunction):
    """Half Multinomial deviance loss with multinomial logit link.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred in (0, 1)**n_classes, i.e. interval with boundaries excluded

    Link:
    y_pred = softmax(raw_prediction)

    Note: Label encoding is built-in, i.e. {0, 1, 2, 3, .., n_classes - 1} is
    mapped to (y_true == k) for k = 0 .. n_classes - 1 which is either 0 or 1.
    """

    # Note that we do not assume memory alignment/contiguity of 2d arrays.
    # There seems to be little benefit in doing so. Benchmarks proofing the
    # opposite are welcome.
    def loss(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[::1] loss_out,         # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C max_value, sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        # We assume n_samples > n_classes. In this case having the inner loop
        # over n_classes is a good default.
        # TODO: If every memoryview is contiguous and raw_prediction is
        #       f-contiguous, can we write a better algo (loops) to improve
        #       performance?
        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]     # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true[i] == k:
                            loss_out[i] -= raw_prediction[i, k]

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]     # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true[i] == k:
                            loss_out[i] -= raw_prediction[i, k]

                    loss_out[i] *= sample_weight[i]

                free(p)

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[::1] loss_out,         # OUT
        G_DTYPE_C[:, :] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C max_value, sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]  # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true [i] == k:
                            loss_out[i] -= raw_prediction[i, k]
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = p_k - (y_true == k)
                        gradient_out[i, k] = p[k] - (y_true[i] == k)

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    max_value = p[n_classes]  # p[-2]
                    sum_exps = p[n_classes + 1]  # p[-1]
                    loss_out[i] = log(sum_exps) + max_value

                    for k in range(n_classes):
                        # label decode y_true
                        if y_true [i] == k:
                            loss_out[i] -= raw_prediction[i, k]
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        gradient_out[i, k] = (p[k] - (y_true[i] == k)) * sample_weight[i]

                    loss_out[i] *= sample_weight[i]

                free(p)

        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[:, :] gradient_out,    # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = y_pred_k - (y_true == k)
                        gradient_out[i, k] = p[k] - (y_true[i] == k)

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        gradient_out[i, k] = (p[k] - (y_true[i] == k)) * sample_weight[i]

                free(p)

        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[:, :] gradient_out,    # OUT
        G_DTYPE_C[:, :] hessian_out,     # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C sum_exps
            Y_DTYPE_C* p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # hessian_k = p_k * (1 - p_k)
                        # gradient_k = p_k - (y_true == k)
                        gradient_out[i, k] = p[k] - (y_true[i] == k)
                        hessian_out[i, k] = p[k] * (1. - p[k])

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        p[k] /= sum_exps  # p_k = y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        # hessian_k = p_k * (1 - p_k) * sw
                        gradient_out[i, k] = (p[k] - (y_true[i] == k)) * sample_weight[i]
                        hessian_out[i, k] = (p[k] * (1. - p[k])) * sample_weight[i]

                free(p)

        return np.asarray(gradient_out), np.asarray(hessian_out)


    # This method simplifies the implementation of hessp in linear models,
    # i.e. the matrix-vector product of the full hessian, not only of the
    # diagonal (in the classes) approximation as implemented above.
    def gradient_proba(
        self,
        Y_DTYPE_C[::1] y_true,           # IN
        Y_DTYPE_C[:, :] raw_prediction,  # IN
        Y_DTYPE_C[::1] sample_weight,    # IN
        G_DTYPE_C[:, :] gradient_out,    # OUT
        G_DTYPE_C[:, :] proba_out,       # OUT
        int n_threads=1
    ):
        cdef:
            int i, k
            int n_samples = y_true.shape[0]
            int n_classes = raw_prediction.shape[1]
            Y_DTYPE_C sum_exps
            Y_DTYPE_C*  p  # temporary buffer

        if sample_weight is None:
            # inner loop over n_classes
            with nogil, parallel(num_threads=n_threads):
                # Define private buffer variables as each thread might use its
                # own.
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        proba_out[i, k] = p[k] / sum_exps  # y_pred_k = prob of class k
                        # gradient_k = y_pred_k - (y_true == k)
                        gradient_out[i, k] = proba_out[i, k] - (y_true[i] == k)

                free(p)
        else:
            with nogil, parallel(num_threads=n_threads):
                p = <Y_DTYPE_C *> malloc(sizeof(Y_DTYPE_C) * (n_classes + 2))

                for i in prange(n_samples, schedule='static'):
                    sum_exp_minus_max(i, raw_prediction, p)
                    sum_exps = p[n_classes + 1]  # p[-1]

                    for k in range(n_classes):
                        proba_out[i, k] = p[k] / sum_exps  # y_pred_k = prob of class k
                        # gradient_k = (p_k - (y_true == k)) * sw
                        gradient_out[i, k] = (proba_out[i, k] - (y_true[i] == k)) * sample_weight[i]

                free(p)

        return np.asarray(gradient_out), np.asarray(proba_out)
