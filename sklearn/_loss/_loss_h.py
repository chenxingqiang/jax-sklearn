from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, raw_prediction):
        pass

    @abstractmethod
    def gradient(self, y_true, raw_prediction):
        pass

    @abstractmethod
    def grad_hess(self, y_true, raw_prediction):
        pass


class HalfSquaredError(LossFunction):
    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass


class AbsoluteError(LossFunction):
    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass


class PinballLoss(LossFunction):
    def __init__(self, quantile):
        self.quantile = quantile

    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass


class HalfPoissonLoss(LossFunction):
    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass


class HalfGammaLoss(LossFunction):
    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass


class HalfTweedieLoss(LossFunction):
    def __init__(self, power):
        self.power = power

    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass


class HalfTweedieLossIdentity(LossFunction):
    def __init__(self, power):
        self.power = power

    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass


class HalfBinomialLoss(LossFunction):
    def loss(self, y_true, raw_prediction):
        pass

    def gradient(self, y_true, raw_prediction):
        pass

    def grad_hess(self, y_true, raw_prediction):
        pass
