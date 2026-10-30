import json
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

from periapsis.data import AstrometryData, GaiaData, RadialVelocityData
from periapsis.prior import FixedPrior
from periapsis.stats import stat_funcs

ASTRO_UNITS = {"t": u.yr, "x": u.rad, "y": u.rad, "x_err": u.rad, "y_err": u.rad}
RV_UNITS = {"t": u.yr, "rv": u.AU / u.yr, "rv_err": u.AU / u.yr}


class SubscriptableNamespace(SimpleNamespace):
    """A SimpleNamespace that supports subscriptable access to attributes."""

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return self[key[0]]
        return None if getattr(self, "samples", None) is None else self.samples[key]


class FakeOrbit:
    created = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.derived_params = parameters
        type(self).created.append(self)

    def gaia_astrometry(self, t, spsi, cpsi, plx_fac, system=None):
        return np.zeros_like(t, dtype=float)


def make_results(**overrides):
    arguments = {
        "MAP_params": {"chi2": 8.0},
        "median_params": {"chi2": 10.0},
        "samples": {},
        "sampled_param_names": [],
        "priors": {},
        "null_hypothesis": {"chi2": 20.0, "dof": 7},
        "backend": None,
        "unit_system": None,
    }
    arguments.update(overrides)
    return SimpleNamespace(**arguments)


def install_fake_model_builders(monkeypatch):
    FakeOrbit.created = []
    monkeypatch.setattr(stat_funcs, "Orbit", FakeOrbit)

    def build_model(results, parameters):
        fixed = {
            name: prior.value
            for name, prior in results.priors.items()
            if isinstance(prior, FixedPrior)
        }
        return FakeOrbit(**parameters, **fixed)

    monkeypatch.setattr(stat_funcs, "_build_model", build_model, raising=False)


def test_json_numpy_serializes_numpy_arrays_and_scalars():
    assert stat_funcs._json_numpy(np.array([1, 2])) == [1, 2]
    assert stat_funcs._json_numpy(np.float64(2.5)) == 2.5
    assert stat_funcs._json_numpy(np.int64(4)) == 4


def test_json_numpy_rejects_unsupported_objects():
    with pytest.raises(TypeError, match="not JSON serializable"):
        stat_funcs._json_numpy(object())


def test_credible_interval_summary_uses_standard_one_and_two_sigma_percentiles():
    samples = np.arange(101.0)

    summary = stat_funcs._credible_interval_summary(samples, "UNIT", 0)

    assert summary == pytest.approx(
        {
            "unit": "UNIT",
            "-2sigma": np.percentile(samples, 2.275),
            "-1sigma": np.percentile(samples, 15.865),
            "median": np.percentile(samples, 50),
            "map": samples[0],
            "+1sigma": np.percentile(samples, 84.135),
            "+2sigma": np.percentile(samples, 97.725),
        }
    )


def test_credible_intervals_includes_derived_secondary_mass():
    mass_samples = np.array([1.0, 2.0, 3.0, 4.0])
    results = SubscriptableNamespace(
        param_names=["P"],
        samples={"P": np.arange(4.0), "M2": mass_samples, "minM2": mass_samples + 1.0},
        map_index=0,
        unit_system=None,
    )

    intervals = stat_funcs.credible_intervals(results)

    assert set(intervals) == {"P", "M2"}
    assert intervals["M2"] == pytest.approx(
        stat_funcs._credible_interval_summary(mass_samples, None, 0)
    )


def test_credible_intervals_uses_minimum_mass_when_secondary_mass_is_missing():
    min_mass_samples = np.array([1.0, 2.0, 3.0, 4.0])
    results = SubscriptableNamespace(
        param_names=["P"],
        samples={"P": np.arange(4.0), "minM2": min_mass_samples},
        map_index=0,
        unit_system=None,
    )

    intervals = stat_funcs.credible_intervals(results)

    assert set(intervals) == {"P", "minM2"}
    assert intervals["minM2"] == pytest.approx(
        stat_funcs._credible_interval_summary(min_mass_samples, None, 0)
    )


def test_latex_credible_interval_uses_two_decimal_places():
    summary = {
        "-1sigma": 1.012,
        "median": 1.234,
        "+1sigma": 1.567,
    }

    assert stat_funcs._latex_credible_interval("P", summary) == (
        r"${P} = 1.23^{+0.33}_{-0.22}$"
    )


def test_red_chi2_requires_map_and_median_parameters():
    data = SimpleNamespace(t=np.arange(3.0))

    with pytest.raises(ValueError):
        stat_funcs.red_chi2(
            make_results(MAP_params=None, median_params={"P": 1.0}),
            data,
        )
    with pytest.raises(ValueError):
        stat_funcs.red_chi2(
            make_results(MAP_params={"P": 1.0}, median_params=None),
            data,
        )


def test_red_chi2_falls_back_to_parameter_sets_in_samples(monkeypatch):
    install_fake_model_builders(monkeypatch)
    data = AstrometryData(
        t=np.arange(3.0),
        x=np.zeros(3),
        y=np.zeros(3),
        x_err=1.0,
        y_err=1.0,
        plxf_x=1.0,
        plxf_y=1.0,
        units=ASTRO_UNITS,
        system=1,
    )
    data.chi2 = lambda model, **kwargs: model.parameters["chi2"]
    results = make_results(
        MAP_params=None,
        median_params=None,
        samples={
            "MAP_params": {"chi2": 8.0, "P": 2.0},
            "median_params": {"chi2": 4.0, "P": 3.0},
        },
    )

    result = stat_funcs.red_chi2(results, data)

    assert result == pytest.approx((2.0, 1.0, np.sqrt(2.0), 1.0, 4))


def test_red_chi2_can_build_models_through_the_public_orbit_api(monkeypatch):
    FakeOrbit.created = []
    monkeypatch.setattr(stat_funcs, "Orbit", FakeOrbit)
    data = AstrometryData(
        t=np.arange(2.0),
        x=np.zeros(2),
        y=np.zeros(2),
        x_err=1.0,
        y_err=1.0,
        plxf_x=1.0,
        plxf_y=1.0,
        units=ASTRO_UNITS,
        system=1,
    )
    data.chi2 = lambda model, **kwargs: model.parameters["chi2"]
    results = make_results(
        MAP_params={"chi2": 6.0},
        median_params={"chi2": 3.0},
    )

    result = stat_funcs.red_chi2(results, data)

    assert result == pytest.approx((2.0, 1.0, np.sqrt(2.0), 1.0, 3))


def test_red_chi2_merges_fixed_priors_into_models(monkeypatch):
    install_fake_model_builders(monkeypatch)
    data = AstrometryData(
        t=np.arange(4.0),
        x=np.zeros(4),
        y=np.zeros(4),
        x_err=1.0,
        y_err=1.0,
        plxf_x=1.0,
        plxf_y=1.0,
        units=ASTRO_UNITS,
        system=1,
    )

    def chi2(model, **kwargs):
        assert model.parameters["e"] == 0.25
        return model.parameters["chi2"]

    data.chi2 = chi2
    results = make_results(
        priors={"e": FixedPrior(0.25)},
        MAP_params={"chi2": 12.0, "P": 2.0, "e": 0.25},
        median_params={"chi2": 6.0, "P": 3.0, "e": 0.25},
    )

    reduced_map, reduced_median, uwe_map, uwe_median, dof = (
        stat_funcs.red_chi2(results, data)
    )

    assert dof == 6
    assert reduced_map == pytest.approx(2.0)
    assert reduced_median == pytest.approx(1.0)
    assert uwe_map == pytest.approx(np.sqrt(2.0))
    assert uwe_median == pytest.approx(1.0)


def test_red_chi2_uses_one_observable_per_rv_epoch(monkeypatch):
    install_fake_model_builders(monkeypatch)
    data = RadialVelocityData(
        t=np.arange(5.0),
        rv=np.zeros(5),
        rv_err=1.0,
        units=RV_UNITS,
        system=1,
    )
    data.chi2 = lambda model, **kwargs: model.parameters["chi2"]
    results = make_results(
        MAP_params={"chi2": 6.0, "P": 2.0},
        median_params={"chi2": 3.0, "P": 2.0},
    )

    result = stat_funcs.red_chi2(results, data)

    assert result == pytest.approx((2.0, 1.0, np.sqrt(2.0), 1.0, 3))


def test_delta_chi2_requires_map_and_median_parameters():
    data = SimpleNamespace(t=np.arange(3.0))

    with pytest.raises(ValueError):
        stat_funcs.delta_chi2(
            make_results(MAP_params=None),
            data,
        )


def test_delta_chi2_non_gaia_compares_proper_motion_fit(monkeypatch):
    install_fake_model_builders(monkeypatch)
    data = AstrometryData(
        t=np.arange(4.0),
        x=np.zeros(4),
        y=np.zeros(4),
        x_err=1.0,
        y_err=1.0,
        plxf_x=1.0,
        plxf_y=1.0,
        units=ASTRO_UNITS,
        system=1,
    )
    data.chi2 = lambda model, **kwargs: model.parameters["chi2"]
    results = make_results(
        MAP_params={"chi2": 8.0, "P": 2.0},
        median_params={"chi2": 10.0, "P": 3.0},
        null_hypothesis={"chi2": 20.0, "dof": 8},
    )
    sf_calls = []

    class FakeChi2Distribution:
        @staticmethod
        def sf(value, dof):
            sf_calls.append((value, dof))
            return value / 100.0 + dof / 1000.0

    monkeypatch.setattr(stat_funcs, "chi2", FakeChi2Distribution)

    result = stat_funcs.delta_chi2(results, data)

    assert result == pytest.approx((12.0, 10.0, 0.122, 0.102, 3.0))
    assert sf_calls == [(12.0, 2), (10.0, 2)]


def install_all_stats_dependencies(monkeypatch):
    monkeypatch.setattr(
        stat_funcs,
        "red_chi2",
        lambda results, data: (2.0, 1.5, np.sqrt(2.0), np.sqrt(1.5), 7),
    )
    monkeypatch.setattr(
        stat_funcs,
        "delta_chi2",
        lambda results, data: (12.0, 10.0, 0.01, 0.02, 3.0),
    )
    monkeypatch.setattr(
        stat_funcs,
        "credible_intervals",
        lambda results: {
            "P": {
                "-2sigma": 1.0,
                "-1sigma": 2.0,
                "median": 2.5,
                "+1sigma": 3.0,
                "+2sigma": 4.0,
            }
        },
    )


def test_all_stats_combines_summaries_and_emcee_diagnostics(monkeypatch):
    install_all_stats_dependencies(monkeypatch)
    results = make_results(
        backend="emcee",
        Ess=120,
        mean_acceptance_fraction=0.35,
        tau=np.array([4.0, 5.0]),
    )

    stats, fit_results = stat_funcs.all_stats(
        results,
        object(),
        pretty_print=False,
    )

    assert stats["red_chi2_map"] == 2.0
    assert stats["red_chi2_med"] == 1.5
    assert stats["dof"] == 7
    assert stats["delta_chi2_map"] == 12.0
    assert stats["p_value_med"] == 0.02
    assert stats["Ess"] == 120
    assert stats["mean_acceptance_fraction"] == 0.35
    np.testing.assert_allclose(stats["tau"], [4.0, 5.0])
    assert fit_results["MAP_params"] is results.MAP_params


def test_all_stats_omits_sampler_diagnostics_for_non_emcee_backend(monkeypatch):
    install_all_stats_dependencies(monkeypatch)

    stats, _ = stat_funcs.all_stats(
        make_results(backend="ultranest"),
        object(),
        pretty_print=False,
    )

    assert "Ess" not in stats
    assert "tau" not in stats
    assert "mean_acceptance_fraction" not in stats


def test_all_stats_pretty_prints_json(monkeypatch, capsys):
    install_all_stats_dependencies(monkeypatch)

    stat_funcs.all_stats(make_results(), object(), pretty_print=True, indent=2)

    output = capsys.readouterr().out
    assert '"red_chi2_map": 2.0' in output
    assert '"MAP_params"' in output
    assert "=" * 80 in output


def test_all_stats_saves_json_files(monkeypatch, tmp_path):
    install_all_stats_dependencies(monkeypatch)
    results = make_results(
        MAP_params={"P": np.float64(4.0)},
        median_params={"P": np.float64(5.0)},
    )

    stats, fit_results = stat_funcs.all_stats(
        results,
        object(),
        pretty_print=False,
        savepath=tmp_path,
    )

    stats_path = tmp_path / "stats.json"
    fit_path = tmp_path / "fit_results.json"
    assert json.loads(stats_path.read_text()) == json.loads(
        json.dumps(stats, default=stat_funcs._json_numpy)
    )
    assert json.loads(fit_path.read_text()) == json.loads(
        json.dumps(fit_results, default=stat_funcs._json_numpy)
    )
