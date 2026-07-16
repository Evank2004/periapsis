import numpy as np

from periapsis.prior import NormalPrior
from periapsis.prior import LogNormalPrior
from periapsis.prior import UniformPrior


def test_uniform_prior_logpdf_inside_and_outside_bounds():
    prior = UniformPrior(1.0, 3.0)

    assert prior.logpdf(2.0) == -np.log(2.0)
    assert prior.logpdf(0.5) == -np.inf


def test_uniform_prior_sample_stays_within_bounds():
    prior = UniformPrior(-2.0, 5.0)
    random_state = np.random.RandomState(0)

    samples = prior.sample(random_state, size=100)

    assert samples.shape == (100,)
    assert np.all(samples >= -2.0)
    assert np.all(samples <= 5.0)


def test_normal_prior_logpdf_is_maximum_at_the_mean():
    prior = NormalPrior(10.0, 2.0)

    assert np.isfinite(prior.logpdf(10.0))
    assert prior.logpdf(10.0) > prior.logpdf(12.0)


def test_log_normal_prior_logpdf_matches_base_10_density():
    prior = LogNormalPrior(1.0, 0.5)

    expected = -0.5*np.log(2*np.pi*0.5**2) - np.log(10.0*np.log(10))

    assert prior.logpdf(10.0) == expected
    assert prior.logpdf(0.0) == -np.inf


def test_log_normal_prior_sample_is_positive():
    prior = LogNormalPrior(1.0, 0.5)
    random_state = np.random.RandomState(0)

    samples = prior.sample(random_state, size=100)

    assert samples.shape == (100,)
    assert np.all(samples > 0)
