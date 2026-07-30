import numpy as np
import pytest

from periapsis.fitting.results import FitResults, SampledPriors
from periapsis.prior import Bounds, FixedPrior, UniformPrior


class SequencedPrior:
    """Return prescribed batches so rejection sampling is deterministic."""

    def __init__(self, *batches):
        self.batches = [np.asarray(batch, dtype=float) for batch in batches]
        self.calls = []

    def sample(self, random_state, size=1):
        self.calls.append((random_state, size))
        batch = self.batches.pop(0)
        assert batch.shape == (size,)
        return batch


def make_results(**overrides):
    arguments = {
        "param_names": ["P", "e"],
        "P": np.array([2.0, 4.0, 8.0]),
        "e": np.array([0.1, 0.2, 0.3]),
        "priors": {},
    }
    arguments.update(overrides)
    return FitResults(**arguments)


def test_fit_results_separates_metadata_from_parameter_samples():
    raw_sampler = object()
    backend = object()
    map_parameters = {"P": 4.0}
    median_parameters = {"P": 5.0}

    results = FitResults(
        param_names=["P"],
        P=np.array([3.0, 4.0]),
        extra=np.array([9.0, 10.0]),
        raw_sampler=raw_sampler,
        backend=backend,
        fit_method="mcmc",
        MAP_params=map_parameters,
        median_params=median_parameters,
        PM_fit={"chi2": 2.0},
        Single_motion_params={"chi2": 3.0},
        Ess=100,
        tau=5.0,
        mean_acceptance_fraction=0.4,
        lnprob=np.array([-2.0, -1.0]),
    )

    assert results.raw_samples is raw_sampler
    assert results.backend is backend
    assert results.sampler is backend
    assert results.fit_method == "mcmc"
    assert results.MAP_params is map_parameters
    assert results.median_params is median_parameters
    assert results.Ess == 100
    assert results.tau == 5.0
    assert results.mean_acceptance_fraction == 0.4
    np.testing.assert_allclose(results.lnprob, [-2.0, -1.0])
    assert set(results.samples) == {"P", "extra", "param_names"}
    assert results.samples["param_names"] == ["P"]


def test_fit_results_uses_independent_empty_prior_mappings():
    first = FitResults(param_names=["P"], P=np.array([2.0]))
    second = FitResults(param_names=["P"], P=np.array([3.0]))

    first.priors["new"] = FixedPrior(1.0)

    assert second.priors == {}


def test_fit_results_tracks_sampled_fixed_and_covered_parameters():
    results = FitResults(
        param_names=["P"],
        P=np.array([2.0, 4.0]),
        priors={"e": FixedPrior(0.25)},
    )

    assert results.known_params == {"P", "e"}
    assert {"P", "e", "n"} <= results.covered_params


def test_fit_results_getitem_returns_sampled_parameter_array():
    results = make_results()

    assert results["P"] is results.samples["P"]
    np.testing.assert_allclose(results["P"], [2.0, 4.0, 8.0])


def test_fit_results_getitem_returns_fixed_prior_value():
    results = FitResults(
        param_names=["P"],
        P=np.array([2.0, 4.0]),
        priors={"e": FixedPrior(0.4)},
    )

    assert results["e"] == 0.4


def test_fit_results_derives_vector_parameters_from_samples():
    results = FitResults(
        param_names=["P"],
        P=np.array([1.0, 2.0, 4.0]),
    )

    result = results["n"]

    np.testing.assert_allclose(result, [2.0 * np.pi, np.pi, np.pi / 2.0])


def test_fit_results_derives_using_sampled_and_fixed_parameters():
    results = FitResults(
        param_names=["P", "a1"],
        P=np.array([10.0, 20.0]),
        a1=np.array([1.0, 2.0]),
        priors={"M1": FixedPrior(1.0)},
    )

    assert "M2" in results.covered_params
    assert "M2" in results
    assert results["M2"].shape == (2,)
    assert np.all(np.isfinite(results["M2"]))
    assert np.all(results["M2"] > 0.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("P", True),
        ("e", True),
        ("n", True),
        ("distance", False),
        ("not-a-parameter", False),
    ],
)
def test_fit_results_contains_reports_known_and_derivable_names(name, expected):
    results = FitResults(
        param_names=["P"],
        P=np.array([2.0, 3.0]),
        priors={"e": FixedPrior(0.2)},
    )

    assert (name in results) is expected


def test_fit_results_getitem_raises_for_unreachable_parameter():
    results = FitResults(param_names=["P"], P=np.array([2.0, 4.0]))

    with pytest.raises(KeyError):
        _ = results["e"]


def test_fit_results_accepts_explicit_no_priors_and_reports_sampling_error():
    results = FitResults(
        param_names=["P"],
        P=np.array([2.0, 4.0]),
        priors=None,
    )

    with pytest.raises(ValueError, match="No priors"):
        results.sample_priors(np.random.RandomState(0), size=2)


def test_sample_priors_returns_sampled_priors_with_requested_configuration():
    results = FitResults(
        param_names=["P"],
        P=np.array([2.0, 4.0]),
        priors={"P": UniformPrior(1.0, 5.0)},
    )
    rng = np.random.RandomState(3)

    sampled = results.sample_priors(rng, size=6)

    assert isinstance(sampled, SampledPriors)
    assert sampled.priors is results.priors
    assert sampled.param_order == ["P"]
    assert sampled.size == 6
    assert sampled.rng is rng
    assert sampled.sampled_priors.shape == (6, 1)


def test_sampled_priors_preserves_parameter_order_and_rng_sequence():
    priors = {
        "P": UniformPrior(1.0, 5.0),
        "e": UniformPrior(0.0, 0.8),
    }
    actual_rng = np.random.RandomState(5)
    expected_rng = np.random.RandomState(5)
    expected_periods = priors["P"].sample(expected_rng, size=4)
    expected_eccentricities = priors["e"].sample(expected_rng, size=4)

    sampled = SampledPriors(
        priors,
        param_order=["e", "P"],
        size=4,
        rng=actual_rng,
    )

    np.testing.assert_allclose(sampled.sampled_priors[:, 0], expected_eccentricities)
    np.testing.assert_allclose(sampled.sampled_priors[:, 1], expected_periods)


def test_sampled_priors_transforms_prior_parameterization_to_output_order():
    sampled = SampledPriors(
        {"P": FixedPrior(4.0)},
        param_order=["n"],
        size=3,
        rng=np.random.RandomState(0),
    )

    np.testing.assert_allclose(sampled.sampled_priors, np.full((3, 1), np.pi / 2.0))


def test_sampled_priors_rejects_and_resamples_values_outside_derived_bounds():
    rng = np.random.RandomState(0)
    period_prior = SequencedPrior(
        [1.0, 2.0, 4.0],
        [2.0, 2.0],
    )

    sampled = SampledPriors(
        {
            "P": period_prior,
            "n": Bounds(lower=2.0, upper=4.0),
        },
        param_order=["P"],
        size=3,
        rng=rng,
    )

    np.testing.assert_allclose(sampled.sampled_priors[:, 0], [2.0, 2.0, 2.0])
    assert period_prior.calls == [(rng, 3), (rng, 2)]


def test_sampled_priors_warns_once_when_a_bound_cannot_be_applied(capsys):
    sampled = SampledPriors(
        {
            "P": FixedPrior(4.0),
            "e": Bounds(lower=0.0, upper=0.9),
        },
        param_order=["P"],
        size=2,
        rng=np.random.RandomState(0),
    )

    output = capsys.readouterr().out
    assert output.count("cannot be applied") == 1
    np.testing.assert_allclose(sampled.sampled_priors[:, 0], 4.0)


def test_sampled_priors_records_and_warns_about_overconstraints(capsys):
    sampled = SampledPriors(
        {"P": FixedPrior(4.0), "n": FixedPrior(np.pi / 2.0)},
        param_order=["P"],
        size=2,
        rng=np.random.RandomState(0),
    )

    assert sampled.overconstrained_priors == {"P", "n"}
    assert "contradictory" in capsys.readouterr().out


def test_sampled_priors_uses_nan_for_an_unreachable_output_parameter():
    sampled = SampledPriors(
        {"P": FixedPrior(4.0)},
        param_order=["e"],
        size=3,
        rng=np.random.RandomState(0),
    )

    assert np.all(np.isnan(sampled.sampled_priors[:, 0]))


def test_sampled_priors_getitem_returns_direct_column():
    sampled = SampledPriors(
        {"P": UniformPrior(1.0, 2.0)},
        param_order=["P"],
        size=4,
        rng=np.random.RandomState(0),
    )

    result = sampled["P"]

    assert np.shares_memory(result, sampled.sampled_priors)
    np.testing.assert_allclose(result, sampled.sampled_priors[:, 0])


def test_sampled_priors_getitem_derives_vector_parameter():
    sampled = SampledPriors(
        {"P": UniformPrior(1.0, 2.0)},
        param_order=["P"],
        size=4,
        rng=np.random.RandomState(0),
    )

    np.testing.assert_allclose(sampled["n"], 2.0 * np.pi / sampled["P"])


def test_sampled_priors_getitem_can_return_fixed_prior_value():
    sampled = SampledPriors(
        {"P": FixedPrior(4.0)},
        param_order=["n"],
        size=2,
        rng=np.random.RandomState(0),
    )

    assert sampled["P"] == 4.0


def test_sampled_priors_getitem_raises_for_unreachable_parameter():
    sampled = SampledPriors(
        {"P": FixedPrior(4.0)},
        param_order=["P"],
        size=2,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(KeyError):
        _ = sampled["e"]
