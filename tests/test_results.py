import numpy as np

from periapsis.fitting.results import FitResults


def test_add_mass_samples_adds_derived_mass_and_param_name():
    results = FitResults(
        param_names=["P", "a"],
        P=np.array([10.0, 20.0]),
        a=np.array([1.0, 2.0]),
        m1=1.0,
    )

    results.add_mass_samples()

    assert "M2" in results.samples
    assert results.samples["param_names"] == ["P", "a"]
    assert results.M2.shape == (2,)