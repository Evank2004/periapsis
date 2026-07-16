import numpy as np

from periapsis.fitting.fitter import Fitter


class DummyData:
    def __init__(self, t, x, y, x_err, y_err, ref_epoch, mu_x, mu_y):
        self.t = np.asarray(t)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.x_err = np.asarray(x_err)
        self.y_err = np.asarray(y_err)
        self.ref_epoch = ref_epoch
        self.mu_x = mu_x
        self.mu_y = mu_y


class DummyFitter(Fitter):
    def fit(self, data):
        return data


def test_proper_motion_fit_uses_provided_mu_values():
    fitter = DummyFitter()
    data = DummyData(
        t=[0.0, 1.0, 2.0],
        x=[5.0, 7.0, 9.0],
        y=[-3.0, -4.0, -5.0],
        x_err=[1.0, 1.0, 1.0],
        y_err=[1.0, 1.0, 1.0],
        ref_epoch=1.0,
        mu_x=2.0,
        mu_y=-1.0,
    )

    result = fitter._proper_motion_fit(data)

    assert result["params"]["x0"] == 7.0
    assert result["params"]["y0"] == -4.0
    assert result["params"]["mu_x"] == 2.0
    assert result["params"]["mu_y"] == -1.0