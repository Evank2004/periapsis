from types import SimpleNamespace

import numpy as np

from periapsis.stats import stat_funcs


def test_compute_red_chi2_uses_results_and_data_contract(monkeypatch):
    map_params = {"P": 1.0, "e": 0.1}
    med_params = {"P": 2.0, "e": 0.2, "omega": 0.3}
    map_model = object()
    med_model = object()

    results = SimpleNamespace(
        MAP_params=map_params,
        median_params=med_params,
        samples={"MAP_params": map_params, "median_params": med_params},
    )

    data = SimpleNamespace(t=np.arange(5.0))

    def fake_build_model(results_obj, params):
        if params is map_params:
            return map_model
        if params is med_params:
            return med_model
        raise AssertionError("Unexpected parameters passed to _build_model")

    def fake_chi2(model):
        if model is map_model:
            return 10.0
        if model is med_model:
            return 14.0
        raise AssertionError("Unexpected model passed to data.chi2")

    monkeypatch.setattr(stat_funcs, "_build_model", fake_build_model)
    data.chi2 = fake_chi2

    red_chi2_map, red_chi2_med, uwe_map, uwe_med = stat_funcs.red_chi2(results, data)

    assert np.isclose(red_chi2_map, 10.0 / (2 * len(data.t) - len(map_params)))
    assert np.isclose(red_chi2_med, 14.0 / (2 * len(data.t) - len(med_params)))
    assert np.isclose(uwe_map, np.sqrt(10.0 / (2 * len(data.t) - len(map_params))))
    assert np.isclose(uwe_med, np.sqrt(14.0 / (2 * len(data.t) - len(med_params))))

def test_compute_delta_chi2_uses_results_and_data_contract(monkeypatch):
    pm_chi2 = 5.0
    pm_dof = 3
    map_params = {"P": 1.0, "e": 0.1}
    med_params = {"P": 2.0, "e": 0.2, "omega": 0.3}
    map_model = object()
    med_model = object()

    results = SimpleNamespace(
        PM_fit={"chi2": pm_chi2, "dof": pm_dof},
        MAP_params=map_params,
        median_params=med_params,
        samples={"MAP_params": map_params, "median_params": med_params},
    )

    data = SimpleNamespace(t=np.arange(5.0))

    def fake_build_model(results_obj, params):
        if params is map_params:
            return map_model
        if params is med_params:
            return med_model
        raise AssertionError("Unexpected parameters passed to _build_model")

    def fake_chi2(model):
        if model is map_model:
            return 10.0
        if model is med_model:
            return 14.0
        raise AssertionError("Unexpected model passed to data.chi2")

    monkeypatch.setattr(stat_funcs, "_build_model", fake_build_model)
    data.chi2 = fake_chi2

    delta_chi2_map, delta_chi2_med,p_map,p_med = stat_funcs.delta_chi2(results, data)

    assert np.isclose(delta_chi2_map, pm_chi2 - 10.0)
    assert np.isclose(delta_chi2_med, pm_chi2 - 14.0)

def test_credible_intervals_computes_correct_percentiles():
    samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    results = SimpleNamespace(
        param_names=["P"],
        P=samples,
        samples={"P": samples},
    )

    intervals = stat_funcs.credible_intervals(results)

    assert intervals["P"]["-2sigma"] == np.percentile(samples, 2.275)
    assert intervals["P"]["+2sigma"] == np.percentile(samples, 97.725)