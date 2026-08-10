from types import SimpleNamespace

import numpy as np
import pytest

import periapsis.initial as initial_package
import periapsis.initial.astrometry_initial as astrometry_module
import periapsis.initial.gaia_initial as gaia_module
import periapsis.initial.joint_initial as joint_module
import periapsis.initial.rv_initial as rv_module
from periapsis.data import AstrometryData, GaiaData, RadialVelocityData
from periapsis.initial import (
    AstrometryInitialGuess,
    AstrometryLinearInitialGuess,
    GaiaInitialGuess,
    InitialGuess,
    JointInitialGuess,
    RVInitialGuess,
)
from periapsis.prior import FixedPrior


class StubPrior:
    def __init__(self, minimum, maximum, sample_value=None, logpdf_value=0.0):
        self.min = minimum
        self.max = maximum
        self.sample_value = (
            (minimum + maximum) / 2.0
            if sample_value is None
            else sample_value
        )
        self.logpdf_value = logpdf_value
        self.sample_calls = []
        self.logpdf_calls = []

    def sample(self, random_state, size=1):
        self.sample_calls.append((random_state, size))
        return np.full(size, self.sample_value, dtype=float)

    def logpdf(self, value):
        self.logpdf_calls.append(value)
        return self.logpdf_value


class RecordingRNG:
    def __init__(self):
        self.uniform_calls = []
        self.normal_calls = []

    def uniform(self, low, high, size=None):
        self.uniform_calls.append((low, high, size))
        value = (low + high) / 2.0
        if size is None:
            return value
        return np.full(size, value, dtype=float)

    def normal(self, loc=0.0, scale=1.0, size=None):
        self.normal_calls.append((loc, scale, size))
        if size is None:
            return loc
        return loc + scale * np.arange(np.prod(np.atleast_1d(size))).reshape(size)


def make_rv_data():
    return SimpleNamespace(
        t=np.array([0.0, 1.0, 2.0, 3.0]),
        rv=np.array([0.0, 1.0, 0.0, -1.0]),
        rv_err=np.ones(4),
    )


def make_astrometry_data():
    return SimpleNamespace(
        t=np.linspace(0.0, 5.0, 6),
        x=np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0]),
        y=np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
        x_err=np.ones(6),
        y_err=np.ones(6),
    )


def make_gaia_data():
    return SimpleNamespace(
        spsi=np.array([0.0, 1.0, 0.0, -1.0]),
        cpsi=np.array([1.0, 0.0, -1.0, 0.0]),
        plx_fac=np.array([0.2, 0.3, 0.4, 0.5]),
        t=np.array([0.0, 1.0, 2.0, 3.0]),
        x=np.array([1.0, -0.5, 0.25, 0.75]),
        err=np.full(4, 0.1),
    )


def test_initial_package_exports_all_initial_guess_classes():
    assert initial_package.__all__ == [
        "InitialGuess",
        "AstrometryInitialGuess",
        "AstrometryLinearInitialGuess",
        "RVInitialGuess",
        "GaiaInitialGuess",
        "JointInitialGuess",
    ]


@pytest.mark.parametrize(
    "guess_class",
    [
        AstrometryInitialGuess,
        AstrometryLinearInitialGuess,
        RVInitialGuess,
        GaiaInitialGuess,
        JointInitialGuess,
    ],
)
def test_concrete_initial_guess_classes_share_the_base_class(guess_class):
    assert issubclass(guess_class, InitialGuess)


def test_initial_guess_base_class_is_abstract():
    with pytest.raises(TypeError):
        InitialGuess(object(), np.random.RandomState(0))


def test_initial_guess_stores_data_rng_and_priors_without_copying():
    class ConcreteInitialGuess(InitialGuess):
        def get_initial_guess(self, param_order, nwalkers):
            return np.empty((nwalkers, len(param_order)))

    data = object()
    rng = np.random.RandomState(2)
    period_prior = StubPrior(1.0, 10.0)

    guess = ConcreteInitialGuess(data, rng, P=period_prior)

    assert guess.data is data
    assert guess.rng is rng
    assert guess.priors == {"P": period_prior}


def test_rv_bounds_preserve_requested_parameter_order():
    guess = RVInitialGuess(
        make_rv_data(),
        np.random.RandomState(0),
        e=StubPrior(0.0, 0.9),
        P=StubPrior(2.0, 20.0),
    )

    assert guess.Bounds(["P", "e"]) == [(2.0, 20.0), (0.0, 0.9)]


def test_rv_bounds_reject_a_parameter_without_a_prior():
    guess = RVInitialGuess(
        make_rv_data(),
        np.random.RandomState(0),
        P=StubPrior(2.0, 20.0),
    )

    with pytest.raises(ValueError, match="e"):
        guess.Bounds(["P", "e"])


def test_zucker_period_search_uses_prior_bounds_and_best_period(monkeypatch):
    guess = RVInitialGuess(
        make_rv_data(),
        np.random.RandomState(0),
        P=StubPrior(1.0, 8.0),
    )
    logspace_calls = []

    def fake_logspace(low, high, num):
        logspace_calls.append((low, high, num))
        return np.array([1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0])

    monkeypatch.setattr(rv_module.np, "logspace", fake_logspace)
    monkeypatch.setattr(rv_module.np, "argmax", lambda values: 1)

    result = guess.Zucker_pdc(num_freq=3)

    assert result == pytest.approx(4.0)
    assert logspace_calls == [
        (pytest.approx(np.log10(1.0 / 8.0)), pytest.approx(np.log10(1.0)), 3)
    ]


def test_zucker_period_search_uses_default_range_when_no_period_prior(
    monkeypatch,
):
    guess = RVInitialGuess(make_rv_data(), np.random.RandomState(0))
    logspace_calls = []

    def fake_logspace(low, high, num):
        logspace_calls.append((low, high, num))
        return np.array([1.0 / 100.0, 1.0 / 50.0, 1.0 / 25.0, 1.0 / 10.0, 1.0])

    monkeypatch.setattr(rv_module.np, "logspace", fake_logspace)
    monkeypatch.setattr(rv_module.np, "argmax", lambda values: 0)

    result = guess.Zucker_pdc(num_freq=5)

    assert result == pytest.approx(100.0)
    assert logspace_calls == [
        (pytest.approx(np.log10(0.01)), pytest.approx(np.log10(1000.0)), 5)
    ]


def test_rv_negative_log_posterior_combines_chi2_and_priors(monkeypatch):
    data = make_rv_data()
    data.chi2 = lambda model: 8.0
    varying = StubPrior(-10.0, 10.0, logpdf_value=-1.5)
    fixed = FixedPrior(2.0)
    guess = RVInitialGuess(
        data,
        np.random.RandomState(0),
        systemic_velocity=varying,
        P=fixed,
    )
    captured = {}

    class FakeOrbit:
        def __init__(self, **parameters):
            captured.update(parameters)

    monkeypatch.setattr(rv_module, "Orbit", FakeOrbit)

    result = guess.neg_lnlike([3.0, 2.0], data)

    assert captured == {"systemic_velocity": 3.0, "P": 2.0}
    assert varying.logpdf_calls == [3.0]
    assert result == pytest.approx(5.5)


def test_rv_negative_log_posterior_is_negative_infinity_outside_prior(
    monkeypatch,
):
    data = make_rv_data()
    data.chi2 = lambda model: 0.0
    prior = StubPrior(0.0, 1.0, logpdf_value=-np.inf)
    guess = RVInitialGuess(data, np.random.RandomState(0), e=prior)
    monkeypatch.setattr(rv_module, "Orbit", lambda **parameters: object())

    result = guess.neg_lnlike([2.0], data)

    assert result == -np.inf


def test_rv_initial_guess_runs_optimizers_clips_and_transforms(monkeypatch):
    data = make_rv_data()
    rng = RecordingRNG()
    period_prior = StubPrior(1.0, 10.0, sample_value=7.0)
    eccentricity_prior = StubPrior(0.0, 0.9, sample_value=0.3)
    guess = RVInitialGuess(
        data,
        rng,
        P=period_prior,
        e=eccentricity_prior,
    )
    monkeypatch.setattr(guess, "Zucker_pdc", lambda: 5.0)
    optimizer_calls = {}

    def fake_differential_evolution(function, bounds, args, maxiter, polish):
        optimizer_calls["global"] = (function, bounds, args, maxiter, polish)
        return SimpleNamespace(x=np.array([6.0, 0.4]))

    def fake_minimize(function, x0, method, args, bounds, options):
        optimizer_calls["local"] = (
            function,
            x0,
            method,
            args,
            bounds,
            options,
        )
        return SimpleNamespace(x=np.array([20.0, -1.0]))

    monkeypatch.setattr(rv_module, "differential_evolution", fake_differential_evolution)
    monkeypatch.setattr(rv_module, "minimize", fake_minimize)

    result = guess.get_initial_guess(["n", "e"], nwalkers=3)

    expected = np.array(
        [
            [2.0 * np.pi / 10.0, 0.0],
            [2.0 * np.pi / 10.0 + 1e-4, 1e-4],
            [2.0 * np.pi / 10.0 + 2e-4, 2e-4],
        ]
    )
    np.testing.assert_allclose(result, expected)
    assert result.shape == (3, 2)
    assert eccentricity_prior.sample_calls == [(rng, 1)]
    assert optimizer_calls["global"][1:] == (
        [(1.0, 10.0), (0.0, 0.9)],
        (data,),
        2000,
        False,
    )
    assert optimizer_calls["local"][2:] == (
        "L-BFGS-B",
        (data,),
        [(1.0, 10.0), (0.0, 0.9)],
        {"maxiter": 2000},
    )
    np.testing.assert_array_equal(optimizer_calls["local"][1], [6.0, 0.4])


def test_astrometry_bounds_preserve_order_and_require_every_prior():
    guess = AstrometryInitialGuess(
        make_astrometry_data(),
        np.random.RandomState(0),
        e=StubPrior(0.0, 0.8),
        P=StubPrior(2.0, 9.0),
    )

    assert guess._bounds(["P", "e"]) == [(2.0, 9.0), (0.0, 0.8)]
    with pytest.raises(ValueError, match="a"):
        guess._bounds(["P", "a"])


def test_astrometry_log_likelihood_constructs_orbit_and_uses_chi2(monkeypatch):
    data = make_astrometry_data()
    captured = {}

    class FakeOrbit:
        def __init__(self, **parameters):
            captured.update(parameters)

    data.chi2 = lambda model: 12.0
    monkeypatch.setattr(astrometry_module, "Orbit", FakeOrbit)
    guess = AstrometryInitialGuess(data, np.random.RandomState(0))

    result = guess.ln_like({"P": 4.0, "e": 0.2}, data)

    assert captured == {"P": 4.0, "e": 0.2}
    assert result == pytest.approx(-6.0)


def test_astrometry_log_prior_sums_terms_and_skips_fixed_values():
    varying = StubPrior(-5.0, 5.0, logpdf_value=-1.25)
    fixed = FixedPrior(3.0)
    guess = AstrometryInitialGuess(
        make_astrometry_data(),
        np.random.RandomState(0),
    )

    result = guess.ln_prior(
        {"x": 2.0, "fixed": 3.0},
        {"x": varying, "fixed": fixed},
    )

    assert result == pytest.approx(-1.25)
    assert varying.logpdf_calls == [2.0]


def test_astrometry_log_prior_handles_impossible_and_missing_values():
    impossible = StubPrior(0.0, 1.0, logpdf_value=-np.inf)
    guess = AstrometryInitialGuess(
        make_astrometry_data(),
        np.random.RandomState(0),
    )

    assert guess.ln_prior({"e": 2.0}, {"e": impossible}) == -np.inf
    with pytest.raises(ValueError, match="e"):
        guess.ln_prior({"e": 0.2}, {})


def test_astrometry_negative_log_posterior_maps_parameter_order(monkeypatch):
    guess = AstrometryInitialGuess(
        make_astrometry_data(),
        np.random.RandomState(0),
    )
    seen = []

    def fake_prior(parameters, priors):
        seen.append(("prior", parameters, priors))
        return -2.0

    def fake_likelihood(parameters, data):
        seen.append(("likelihood", parameters, data))
        return -3.0

    monkeypatch.setattr(guess, "ln_prior", fake_prior)
    monkeypatch.setattr(guess, "ln_like", fake_likelihood)
    priors = {"e": object(), "P": object()}

    result = guess.neg_lnlike(
        [0.2, 5.0],
        guess.data,
        priors,
        ["e", "P"],
    )

    assert result == pytest.approx(5.0)
    assert seen == [
        ("prior", {"e": 0.2, "P": 5.0}, priors),
        ("likelihood", {"e": 0.2, "P": 5.0}, guess.data),
    ]


def test_lomb_scargle_combines_both_coordinates_and_uses_prior_range(
    monkeypatch,
):
    data = make_astrometry_data()
    guess = AstrometryInitialGuess(
        data,
        np.random.RandomState(0),
        P=StubPrior(2.0, 10.0),
    )
    frequencies_seen = []
    best_index = 250

    class FakeLombScargle:
        def __init__(self, times, values, errors):
            assert times is data.t
            self.coordinate = "x" if values is data.x else "y"
            expected_errors = data.x_err if self.coordinate == "x" else data.y_err
            assert errors is expected_errors

        def power(self, frequencies):
            frequencies_seen.append(frequencies.copy())
            power = np.zeros_like(frequencies)
            power[best_index] = 3.0 if self.coordinate == "x" else 4.0
            return power

        def model_parameters(self, frequency):
            assert frequency == pytest.approx(frequencies_seen[0][best_index])
            if self.coordinate == "x":
                return np.array([0.0, 3.0, 4.0])
            return np.array([0.0, 5.0, 12.0])

    monkeypatch.setattr(astrometry_module, "LombScargle", FakeLombScargle)

    axis_guess, period_guess = guess.lomb_scargle()

    expected_frequency = np.linspace(0.1, 0.5, 100000)[best_index]
    assert axis_guess == pytest.approx(np.hypot(5.0, 13.0))
    assert period_guess == pytest.approx(1.0 / expected_frequency)
    assert len(frequencies_seen) == 2
    np.testing.assert_allclose(frequencies_seen[0], frequencies_seen[1])
    assert frequencies_seen[0][0] == pytest.approx(0.1)
    assert frequencies_seen[0][-1] == pytest.approx(0.5)


def test_lomb_scargle_uses_default_period_range_without_a_period_prior(
    monkeypatch,
):
    data = make_astrometry_data()
    guess = AstrometryInitialGuess(data, np.random.RandomState(0))
    ranges = []

    class FakeLombScargle:
        def __init__(self, times, values, errors):
            pass

        def power(self, frequencies):
            ranges.append((frequencies[0], frequencies[-1]))
            return np.ones_like(frequencies)

        def model_parameters(self, frequency):
            return np.array([0.0, 1.0, 0.0])

    monkeypatch.setattr(astrometry_module, "LombScargle", FakeLombScargle)

    with pytest.raises(ValueError, match="Computing a periodogram requires a direct prior on the period 'P'"):
        guess.lomb_scargle()


def test_astrometry_initial_guess_runs_optimizers_clips_and_transforms(
    monkeypatch,
):
    data = make_astrometry_data()
    rng = RecordingRNG()
    axis_prior = StubPrior(1.0, 10.0, sample_value=4.0)
    period_prior = StubPrior(2.0, 8.0, sample_value=5.0)
    eccentricity_prior = StubPrior(0.0, 0.9, sample_value=0.3)
    guess = AstrometryInitialGuess(
        data,
        rng,
        a=axis_prior,
        P=period_prior,
        e=eccentricity_prior,
    )
    monkeypatch.setattr(guess, "lomb_scargle", lambda: (3.0, 6.0))
    calls = {}

    def fake_differential_evolution(function, bounds, args, maxiter, polish):
        calls["global"] = (function, bounds, args, maxiter, polish)
        return SimpleNamespace(x=np.array([3.0, 6.0, 0.3]))

    def fake_minimize(function, x0, method, args, bounds, constraints, options):
        calls["local"] = (function, x0, method, args, bounds, constraints, options)
        return SimpleNamespace(x=np.array([20.0, 20.0, -1.0]))

    monkeypatch.setattr(
        astrometry_module,
        "differential_evolution",
        fake_differential_evolution,
    )
    monkeypatch.setattr(astrometry_module, "minimize", fake_minimize)

    result = guess.get_initial_guess(["P", "e"], nwalkers=2)

    expected = np.array([[8.0, 0.0], [8.0008, 0.0000]])
    np.testing.assert_allclose(result, expected)
    assert eccentricity_prior.sample_calls == [(rng, 1)]
    assert calls["global"][1:] == (
        [(1.0, 10.0), (2.0, 8.0), (0.0, 0.9)],
        (data, guess.priors, ["a", "P", "e"]),
        2000,
        False,
    )
    assert calls["local"][2:] == (
        "SLSQP",
        (data, guess.priors, ["a", "P", "e"]),
        [(1.0, 10.0), (2.0, 8.0), (0.0, 0.9)],
        [],
        {"maxiter": 2000},
    )


def test_delisle_periodogram_uses_prior_range_and_highest_power(monkeypatch):
    data = make_gaia_data()
    guess = GaiaInitialGuess(
        data,
        np.random.RandomState(0),
        P=StubPrior(2.0, 8.0),
    )
    matrices = []
    orbital_chi2 = iter([90.0, 40.0, 80.0])

    def fake_periodogram_helper(matrix, values, errors):
        assert values is data.x
        assert errors is data.err
        matrices.append(matrix.copy())
        if matrix.shape[1] == 5:
            return np.zeros(5), 100.0
        return np.zeros(9), next(orbital_chi2)

    monkeypatch.setattr(
        gaia_module,
        "_helper_for_periodogram",
        fake_periodogram_helper,
    )

    period, power = guess.Delisle_periodogram(num_freq=3)

    assert period == pytest.approx(4.0)
    assert power == pytest.approx(0.6)
    assert [matrix.shape for matrix in matrices] == [(4, 5), (4, 9), (4, 9), (4, 9)]
    np.testing.assert_allclose(
        matrices[0],
        np.column_stack(
            [
                data.spsi,
                data.cpsi,
                data.plx_fac,
                data.spsi * data.t,
                data.cpsi * data.t,
            ]
        ),
    )


def test_delisle_periodogram_uses_default_period_range(monkeypatch):
    data = make_gaia_data()
    guess = GaiaInitialGuess(data, np.random.RandomState(0))
    orbital_calls = 0

    def fake_periodogram_helper(matrix, values, errors):
        nonlocal orbital_calls
        if matrix.shape[1] == 5:
            return None, 10.0
        orbital_calls += 1
        return None, float(orbital_calls)

    monkeypatch.setattr(
        gaia_module,
        "_helper_for_periodogram",
        fake_periodogram_helper,
    )

    period, _power = guess.Delisle_periodogram(num_freq=2)

    assert period == pytest.approx(100.0)


def test_gaia_initial_guess_uses_periodogram_and_adds_optional_jitter(
    monkeypatch,
):
    rng = RecordingRNG()
    guess = GaiaInitialGuess(
        make_gaia_data(),
        rng,
        P=StubPrior(2.0, 8.0),
        e=StubPrior(0.1, 0.5),
        Tp=StubPrior(10.0, 14.0),
    )
    monkeypatch.setattr(guess, "Delisle_periodogram", lambda: (4.0, 0.8))
    monkeypatch.setattr(
        gaia_module.np.random,
        "randn",
        lambda *shape: np.zeros(shape),
    )

    result = guess.get_initial_guess(
        ["P", "e", "Tp", "jitter"],
        nwalkers=3,
    )

    np.testing.assert_allclose(
        result,
        np.tile([4.0, 0.3, 12.0, 0.005], (3, 1)),
        rtol=0.5
    )
    assert rng.uniform_calls == [(0.1, 0.5, None), (10.0, 14.0, None)]


def test_gaia_initial_guess_respects_arbitrary_parameter_order(monkeypatch):
    guess = GaiaInitialGuess(
        make_gaia_data(),
        RecordingRNG(),
        P=StubPrior(2.0, 8.0),
        e=StubPrior(0.1, 0.5),
        Tp=StubPrior(10.0, 14.0),
    )
    monkeypatch.setattr(guess, "Delisle_periodogram", lambda: (4.0, 0.8))
    monkeypatch.setattr(
        gaia_module.np.random,
        "randn",
        lambda *shape: np.zeros(shape),
    )

    result = guess.get_initial_guess(["e", "Tp", "P"], nwalkers=2)

    # Check that parameters are in the requested order
    np.testing.assert_allclose(result, np.tile([0.3, 12.0, 4.0], (2, 1)), rtol=0.5)


def test_gaia_walker_scatter_uses_the_supplied_rng(monkeypatch):
    rng = RecordingRNG()
    guess = GaiaInitialGuess(
        make_gaia_data(),
        rng,
        P=StubPrior(2.0, 8.0),
        e=StubPrior(0.1, 0.5),
        Tp=StubPrior(10.0, 14.0),
    )
    monkeypatch.setattr(guess, "Delisle_periodogram", lambda: (4.0, 0.8))

    def reject_global_rng(*shape):
        raise AssertionError("walker scatter must use the injected RNG")

    monkeypatch.setattr(gaia_module.np.random, "randn", reject_global_rng)

    result = guess.get_initial_guess(["P", "e", "Tp"], nwalkers=2)

    assert result.shape == (2, 3)
    assert rng.normal_calls


def test_joint_initial_guess_dispatches_each_supported_data_type(monkeypatch):
    gaia_data = object.__new__(GaiaData)
    rv_data = object.__new__(RadialVelocityData)
    astrometry_data = object.__new__(AstrometryData)
    rng = np.random.RandomState(0)
    prior = StubPrior(1.0, 2.0)
    created = []

    def factory(kind):
        def make(data, supplied_rng, **priors):
            child = SimpleNamespace(kind=kind, data=data)
            created.append((kind, data, supplied_rng, priors, child))
            return child

        return make

    monkeypatch.setattr(joint_module, "GaiaInitialGuess", factory("gaia"))
    monkeypatch.setattr(joint_module, "RVInitialGuess", factory("rv"))
    monkeypatch.setattr(
        joint_module,
        "AstrometryInitialGuess",
        factory("astrometry"),
    )

    guess = JointInitialGuess(
        SimpleNamespace(datas=[gaia_data, rv_data, astrometry_data]),
        rng,
        P=prior,
    )

    assert [child.kind for child in guess.initial_guesses] == [
        "gaia",
        "rv",
        "astrometry",
    ]
    assert [entry[1] for entry in created] == [
        gaia_data,
        rv_data,
        astrometry_data,
    ]
    assert all(entry[2] is rng for entry in created)
    assert all(entry[3] == {"P": prior} for entry in created)


def test_joint_initial_guess_rejects_unsupported_data():
    with pytest.raises(ValueError, match="Unsupported data type"):
        JointInitialGuess(
            SimpleNamespace(datas=[object()]),
            np.random.RandomState(0),
        )


def test_joint_initial_guess_averages_child_guesses_and_forwards_arguments():
    calls = []

    class FakeChild:
        def __init__(self, result):
            self.result = result

        def get_initial_guess(self, param_order, nwalkers):
            calls.append((param_order, nwalkers))
            return self.result

    guess = JointInitialGuess(
        SimpleNamespace(datas=[]),
        np.random.RandomState(0),
    )
    guess.initial_guesses = [
        FakeChild(np.array([[1.0, 2.0], [3.0, 4.0]])),
        FakeChild(np.array([[5.0, 6.0], [7.0, 8.0]])),
    ]

    result = guess.get_initial_guess(["P", "e"], nwalkers=2)

    np.testing.assert_allclose(result, [[3.0, 4.0], [5.0, 6.0]])
    assert calls == [(["P", "e"], 2), (["P", "e"], 2)]
