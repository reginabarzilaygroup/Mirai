import numpy as np


class MiraiCalibrator:
    """
    A class to represent a calibrator for Mirai prediction models.
    Behavior and coefficients are taken from the sklearn.calibration.CalibratedClassifierCV class.
    Make a custom class to avoid sklearn versioning issues.
    """

    def __init__(self, base_estimator_slope, base_estimator_offset, calibrator_slope, calibrator_offset):
        self.base_estimator_slope = base_estimator_slope
        self.base_estimator_offset = base_estimator_offset
        self.calibrator_slope = calibrator_slope
        self.calibrator_offset = calibrator_offset

    def predict_proba(self, X, expand=True):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_probabilities,)
            The input probabilities to recalibrate.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by
            lexicographic order.
        """
        _y = self.base_estimator_slope * X + self.base_estimator_offset
        _y = self.calibrator_slope * _y + self.calibrator_offset
        pos_prob = 1./(1.+np.exp(_y))
        if expand:
            return np.array([1.-pos_prob, pos_prob])
        else:
            return pos_prob


def from_sk_calibrator(sk_calibrator: "sklearn.calibration.CalibratedClassifierCV"):
    """
    Convert a sklearn.calibration.CalibratedClassifierCV object to a MiraiCalibrator object.

    This method was used to create a MiraiCalibrator object from a
    sklearn.calibration.CalibratedClassifierCV object, here for posterity.
    """
    import sklearn
    base_estimator = sk_calibrator.base_estimator
    assert len(sk_calibrator.calibrated_classifiers_) == 1, "Only one calibrated classifier is supported."
    assert len(sk_calibrator.calibrated_classifiers_[0].calibrators_) == 1, "Only one calibrator is supported."
    calibrator = sk_calibrator.calibrated_classifiers_[0].calibrators_[0]
    return MiraiCalibrator(base_estimator.coef_[0], base_estimator.intercept_[0], calibrator.a_, calibrator.b_)