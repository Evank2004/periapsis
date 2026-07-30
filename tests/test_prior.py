import numpy as np
import pytest

import periapsis.prior as prior_package
from periapsis.prior import (
    Bounds,
    FixedPrior,
    LogNormalPrior,
    LogUniformPrior,
    NormalPrior,
    Prior,
    UniformPrior,
)


def test_prior_package_exports_all_public_prior_classes():
    assert prior_package.__all__ == [
        "Prior",
        "Bounds",
        "UniformPrior",
        "LogUniformPrior",
        "LogNormalPrior",
        "NormalPrior",
        "FixedPrior",
    ]


def test_prior_base_class_is_abstract():
    with pytest.raises(TypeError):
        Prior()


def test_uniform_prior_exposes_bounds_and_normalized_density():
    prior = UniformPrior(1.0, 5.0)

    assert prior.lower_bound == 1.0
    assert prior.upper_bound == 5.0
    assert prior.min == 1.0
    assert prior.max == 5.0
    assert prior.logpdf(1.0) == pytest.approx(-np.log(4.0))
    assert prior.logpdf(3.0) == pytest.approx(-np.log(4.0))
    assert prior.logpdf(5.0) == pytest.approx(-np.log(4.0))


@pytest.mark.parametrize("value", [-1.0, 5.0001, np.inf, -np.inf])
def test_uniform_prior_logpdf_is_negative_infinity_outside_bounds(value):
    prior = UniformPrior(0.0, 5.0)

    assert prior.logpdf(value) == -np.inf


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(1.0, 1.0), (2.0, 1.0)],
)
def test_uniform_prior_requires_strictly_increasing_bounds(lower, upper):
    with pytest.raises(ValueError):
        UniformPrior(lower, upper)


def test_uniform_prior_sampling_is_shaped_bounded_and_reproducible():
    first_rng = np.random.RandomState(4)
    second_rng = np.random.RandomState(4)
    prior = UniformPrior(-2.0, 5.0)

    first = prior.sample(first_rng, size=(3, 4))
    second = prior.sample(second_rng, size=(3, 4))

    assert first.shape == (3, 4)
    assert np.all(first >= -2.0)
    assert np.all(first <= 5.0)
    np.testing.assert_allclose(first, second)


def test_uniform_prior_unit_cube_transform_is_linear_and_vectorized():
    prior = UniformPrior(-2.0, 6.0)
    unit_values = np.array([0.0, 0.25, 0.5, 1.0])

    result = prior.unp(unit_values)

    np.testing.assert_allclose(result, [-2.0, 0.0, 2.0, 6.0])


def test_log_uniform_prior_exposes_bounds_and_normalized_density():
    prior = LogUniformPrior(1.0, 100.0)

    assert prior.lower_bound == prior.min == 1.0
    assert prior.upper_bound == prior.max == 100.0
    assert prior.logpdf(1.0) == pytest.approx(-np.log(np.log(100.0)))
    assert prior.logpdf(10.0) == pytest.approx(
        -np.log(10.0 * np.log(100.0))
    )
    assert prior.logpdf(100.0) == pytest.approx(
        -np.log(100.0 * np.log(100.0))
    )


@pytest.mark.parametrize("value", [0.0, 0.5, 101.0, -1.0])
def test_log_uniform_prior_rejects_values_outside_bounds(value):
    assert LogUniformPrior(1.0, 100.0).logpdf(value) == -np.inf


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(0.0, 10.0), (-1.0, 10.0), (10.0, 10.0), (100.0, 10.0)],
)
def test_log_uniform_prior_requires_positive_increasing_bounds(lower, upper):
    with pytest.raises(ValueError):
        LogUniformPrior(lower, upper)


def test_log_uniform_sampling_is_positive_bounded_and_reproducible():
    prior = LogUniformPrior(0.01, 100.0)
    first_rng = np.random.RandomState(7)
    second_rng = np.random.RandomState(7)

    first = prior.sample(first_rng, size=50)
    second = prior.sample(second_rng, size=50)

    assert first.shape == (50,)
    assert np.all(first >= 0.01)
    assert np.all(first <= 100.0)
    np.testing.assert_allclose(first, second)


def test_log_uniform_unit_cube_transform_is_uniform_in_log10():
    prior = LogUniformPrior(1.0, 10000.0)

    result = prior.unp(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    np.testing.assert_allclose(result, [1.0, 10.0, 100.0, 1000.0, 10000.0])


def test_normal_prior_exposes_effective_bounds_and_density_at_mean():
    prior = NormalPrior(10.0, 2.0)
    expected_peak = -0.5 * np.log(2.0 * np.pi * 2.0**2)

    assert prior.mean == 10.0
    assert prior.std == 2.0
    assert prior.min == -10.0
    assert prior.max == 30.0
    assert prior.logpdf(10.0) == pytest.approx(expected_peak)


def test_normal_prior_logpdf_is_symmetric_about_mean():
    prior = NormalPrior(3.0, 1.5)

    assert prior.logpdf(0.0) == pytest.approx(prior.logpdf(6.0))
    assert prior.logpdf(3.0) > prior.logpdf(4.5)


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_normal_prior_requires_positive_standard_deviation(std):
    with pytest.raises(ValueError):
        NormalPrior(0.0, std)


def test_normal_prior_sampling_uses_supplied_random_state():
    prior = NormalPrior(2.0, 0.5)
    first_rng = np.random.RandomState(9)
    second_rng = np.random.RandomState(9)

    first = prior.sample(first_rng, size=(2, 3))
    second = prior.sample(second_rng, size=(2, 3))

    assert first.shape == (2, 3)
    np.testing.assert_allclose(first, second)


def test_normal_prior_unit_cube_transform_maps_median_to_mean():
    prior = NormalPrior(7.0, 2.0)

    result = prior.unp(np.array([0.5, 0.8413447460685429]))

    np.testing.assert_allclose(result, [7.0, 9.0], rtol=1e-7)


def test_log_normal_prior_exposes_effective_bounds():
    prior = LogNormalPrior(1.0, 0.5)

    assert prior.mean == 1.0
    assert prior.std == 0.5
    assert prior.min == pytest.approx(1e-4)
    assert prior.max == pytest.approx(1e6)


def test_log_normal_prior_logpdf_matches_base_10_density():
    prior = LogNormalPrior(1.0, 0.5)
    value = 10.0
    expected = (
        -0.5 * np.log(2.0 * np.pi * 0.5**2)
        - np.log(value * np.log(10.0))
    )

    assert prior.logpdf(value) == pytest.approx(expected)
    assert prior.logpdf(0.0) == -np.inf
    assert prior.logpdf(-1.0) == -np.inf


@pytest.mark.parametrize("std", [0.0, -0.5])
def test_log_normal_prior_requires_positive_standard_deviation(std):
    with pytest.raises(ValueError):
        LogNormalPrior(1.0, std)


def test_log_normal_prior_sampling_is_positive_and_reproducible():
    prior = LogNormalPrior(1.0, 0.5)
    first_rng = np.random.RandomState(11)
    second_rng = np.random.RandomState(11)

    first = prior.sample(first_rng, size=30)
    second = prior.sample(second_rng, size=30)

    assert first.shape == (30,)
    assert np.all(first > 0.0)
    np.testing.assert_allclose(first, second)


def test_log_normal_unit_cube_transform_maps_median_to_ten_power_mean():
    prior = LogNormalPrior(2.0, 0.5)

    assert prior.unp(0.5) == pytest.approx(100.0)


def test_fixed_prior_exposes_degenerate_bounds():
    prior = FixedPrior(3.5)

    assert prior.value == 3.5
    assert prior.min == 3.5
    assert prior.max == 3.5


def test_fixed_prior_sampling_returns_requested_shape_and_value():
    prior = FixedPrior(-2.5)

    samples = prior.sample(np.random.RandomState(0), size=(2, 3))

    assert samples.shape == (2, 3)
    np.testing.assert_allclose(samples, -2.5)


def test_fixed_prior_logpdf_is_not_a_sampled_density():
    with pytest.raises(NotImplementedError, match="fixed"):
        FixedPrior(2.0).logpdf(2.0)


def test_fixed_prior_unit_cube_transform_returns_fixed_value():
    unit_values = np.array([0.0, 0.25, 1.0])

    result = FixedPrior(4.25).unp(unit_values)

    np.testing.assert_allclose(result, [4.25, 4.25, 4.25])


def test_bounds_requires_at_least_one_limit():
    with pytest.raises(ValueError):
        Bounds()


@pytest.mark.parametrize(
    ("prior", "inside", "outside"),
    [
        (Bounds(lower=1.0), 1.0, 0.999),
        (Bounds(upper=3.0), 3.0, 3.001),
        (Bounds(lower=1.0, upper=3.0), 2.0, 4.0),
    ],
)
def test_bounds_logpdf_is_zero_inside_and_negative_infinity_outside(
    prior,
    inside,
    outside,
):
    assert prior.logpdf(inside) == 0.0
    assert prior.logpdf(outside) == -np.inf


def test_bounds_requires_increasing_two_sided_limits():
    with pytest.raises(ValueError):
        Bounds(lower=3.0, upper=1.0)


def test_bounds_cannot_be_sampled_or_unit_cube_transformed():
    prior = Bounds(lower=0.0, upper=1.0)

    with pytest.raises(NotImplementedError, match="sampling"):
        prior.sample(np.random.RandomState(0), size=2)
    with pytest.raises(NotImplementedError, match="unp"):
        prior.unp(np.array([0.25, 0.75]))
