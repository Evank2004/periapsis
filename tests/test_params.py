import numpy as np
import pytest

from periapsis.params import (
    build_transform_function,
    build_transform_functions,
    covered_parameters,
    overconstrained_parameters,
    shortest_path,
    uncovered_parameters,
)


TWO_PI = 2.0 * np.pi
GRAVITATIONAL_CONSTANT = 4.0 * np.pi**2


def transform(known, target, **values):
    """Convenience wrapper for exercising transforms through the public API."""
    return build_transform_function(known, target)(**values)


def test_empty_known_parameters_cover_nothing():
    assert covered_parameters(set()) == set()


def test_covered_parameters_include_inputs_and_transitive_derivations():
    known = {"P", "a"}

    covered = covered_parameters(known)

    assert known <= covered
    assert {"n", "Mtot", "mu"} <= covered


def test_covered_parameters_do_not_claim_unreachable_parameters():
    covered = covered_parameters({"P"})

    assert {"P", "n"} <= covered
    assert "e" not in covered
    assert "parallax" not in covered
    assert "dx" not in covered


def test_covered_parameters_accept_any_iterable_without_mutating_it():
    known = ["P", "a"]

    covered_parameters(known)

    assert known == ["P", "a"]


def test_covered_parameters_returns_an_independent_mutable_set():
    first = covered_parameters({"P"})
    first.add("not-a-parameter")

    second = covered_parameters({"P"})

    assert "not-a-parameter" not in second


def test_uncovered_parameters_are_disjoint_from_covered_parameters():
    covered = covered_parameters({"P", "parallax"})
    uncovered = uncovered_parameters({"P", "parallax"})

    assert covered.isdisjoint(uncovered)
    assert {"P", "n", "parallax", "distance"} <= covered
    assert {"e", "i", "dx", "systemic_velocity"} <= uncovered


def test_shortest_path_is_empty_when_target_is_already_known():
    assert shortest_path({"P"}, "P") == []


def test_shortest_path_contains_an_executable_direct_step():
    path = shortest_path({"parallax"}, "distance")

    assert len(path) == 1
    inputs, function, outputs = path[0]
    assert inputs == ("parallax",)
    assert callable(function)
    assert outputs == ("distance",)


def test_shortest_path_is_minimal_and_topologically_ordered():
    known = {"P", "a"}
    path = shortest_path(known, "mu")
    available = set(known)

    assert len(path) == 2
    for inputs, function, outputs in path:
        assert set(inputs) <= available
        assert callable(function)
        available.update(outputs)
    assert "mu" in available


def test_shortest_path_raises_for_an_unreachable_target():
    with pytest.raises(KeyError):
        shortest_path({"P"}, "e")


def test_single_transform_returns_an_explicit_target_unchanged():
    function = build_transform_function({"P", "n"}, "n")

    assert function(P=4.0, n=123.0) == 123.0


def test_single_transform_uses_only_relevant_runtime_arguments():
    function = build_transform_function({"P", "unused-name"}, "n")

    assert function(P=4.0) == pytest.approx(np.pi / 2.0)


def test_single_transform_ignores_extra_runtime_arguments():
    function = build_transform_function({"P"}, "n")

    assert function(P=4.0, extra=99.0) == pytest.approx(np.pi / 2.0)


def test_single_transform_reports_a_missing_required_value():
    function = build_transform_function({"P"}, "n")

    with pytest.raises(KeyError):
        function()


def test_build_transform_function_raises_for_an_unreachable_target():
    with pytest.raises(KeyError):
        build_transform_function({"P"}, "e")


def test_multi_transform_preserves_target_order_and_removes_duplicates():
    function = build_transform_functions(
        {"P", "a"}, ["mu", "n", "Mtot", "n"]
    )

    result = function(P=2.0, a=4.0)

    assert list(result) == ["mu", "n", "Mtot"]
    assert result["Mtot"] == pytest.approx(16.0)
    assert result["mu"] == pytest.approx(16.0 * GRAVITATIONAL_CONSTANT)
    assert result["n"] == pytest.approx(np.pi)


def test_multi_transform_returns_explicit_values_in_preference_to_derivations():
    function = build_transform_functions(
        {"P", "n", "Mtot", "a"}, ["n", "Mtot"]
    )

    result = function(P=2.0, n=123.0, Mtot=456.0, a=4.0)

    assert result == {"n": 123.0, "Mtot": 456.0}


def test_multi_transform_with_no_targets_returns_an_empty_mapping():
    function = build_transform_functions({"P"}, [])

    assert function(P=2.0) == {}


def test_build_transform_functions_raises_if_any_target_is_unreachable():
    with pytest.raises(KeyError):
        build_transform_functions({"P"}, ["n", "e"])


def test_period_semimajor_axis_and_total_mass_obey_keplers_third_law():
    assert transform({"P", "a"}, "Mtot", P=2.0, a=4.0) == pytest.approx(16.0)
    assert transform({"Mtot", "a"}, "P", Mtot=16.0, a=4.0) == pytest.approx(2.0)
    assert transform({"Mtot", "P"}, "a", Mtot=16.0, P=2.0) == pytest.approx(4.0)


def test_period_mean_motion_and_gravitational_parameter_round_trip():
    n = transform({"P"}, "n", P=8.0)
    period = transform({"n"}, "P", n=n)
    mu = transform({"Mtot"}, "mu", Mtot=3.0)
    total_mass = transform({"mu"}, "Mtot", mu=mu)

    assert n == pytest.approx(np.pi / 4.0)
    assert period == pytest.approx(8.0)
    assert mu == pytest.approx(3.0 * GRAVITATIONAL_CONSTANT)
    assert total_mass == pytest.approx(3.0)


def test_period_to_mean_motion_is_vectorized():
    periods = np.array([1.0, 2.0, 4.0])

    result = transform({"P"}, "n", P=periods)

    np.testing.assert_allclose(result, [TWO_PI, np.pi, np.pi / 2.0])
    np.testing.assert_array_equal(periods, [1.0, 2.0, 4.0])


def test_ellipse_shape_parameters_are_derived_together():
    function = build_transform_functions(
        {"a", "e"}, ["b", "p", "r_p", "r_a"]
    )

    result = function(a=5.0, e=0.6)

    assert result == pytest.approx(
        {"b": 4.0, "p": 3.2, "r_p": 2.0, "r_a": 8.0}
    )


@pytest.mark.parametrize(
    ("known", "values"),
    [
        ({"r_a", "r_p"}, {"r_a": 8.0, "r_p": 2.0}),
        ({"r_p", "e"}, {"r_p": 2.0, "e": 0.6}),
        ({"r_a", "e"}, {"r_a": 8.0, "e": 0.6}),
        ({"r_a", "a"}, {"r_a": 8.0, "a": 5.0}),
        ({"r_p", "a"}, {"r_p": 2.0, "a": 5.0}),
        ({"p", "e"}, {"p": 3.2, "e": 0.6}),
    ],
)
def test_alternative_ellipse_parameterizations_recover_a_and_e(known, values):
    function = build_transform_functions(known, ["a", "e"])

    result = function(**values)

    assert result == pytest.approx({"a": 5.0, "e": 0.6})


def test_component_specific_ellipse_parameters_use_the_shared_eccentricity():
    function = build_transform_functions(
        {"a1", "e"}, ["b1", "p1", "r_p1", "r_a1"]
    )

    result = function(a1=2.5, e=0.6)

    assert result == pytest.approx(
        {"b1": 2.0, "p1": 1.6, "r_p1": 1.0, "r_a1": 4.0}
    )


def test_inclination_converts_to_sine_and_cosine_and_back():
    inclination = 2.2
    trig = build_transform_functions({"i"}, ["sini", "cosi"])(i=inclination)
    recovered = transform(
        {"sini", "cosi"},
        "i",
        sini=trig["sini"],
        cosi=trig["cosi"],
    )

    assert trig == pytest.approx(
        {"sini": np.sin(inclination), "cosi": np.cos(inclination)}
    )
    assert recovered == pytest.approx(inclination)


def test_longitude_of_periastron_is_wrapped_to_one_revolution():
    piomega = transform(
        {"omega", "Omega"},
        "piomega",
        omega=1.5 * np.pi,
        Omega=np.pi,
    )
    recovered = transform(
        {"piomega", "Omega"},
        "omega",
        piomega=piomega,
        Omega=np.pi,
    )

    assert piomega == pytest.approx(np.pi / 2.0)
    assert recovered == pytest.approx(1.5 * np.pi)


def test_primary_and_secondary_periastron_arguments_differ_by_pi():
    result = build_transform_functions(
        {"omega"}, ["omega1", "omega2"]
    )(omega=1.75 * np.pi)

    assert result["omega1"] == pytest.approx(0.75 * np.pi)
    assert result["omega2"] == pytest.approx(1.75 * np.pi)


def test_thiele_innes_constants_have_expected_axis_aligned_values():
    constants = build_transform_functions(
        {"a", "cosi", "omega", "Omega"}, ["A", "B", "F", "G"]
    )(a=2.0, cosi=0.5, omega=0.0, Omega=0.0)

    assert constants == pytest.approx({"A": 2.0, "B": 0.0, "F": 0.0, "G": 1.0})


def test_axis_aligned_thiele_innes_constants_round_trip():
    elements = build_transform_functions(
        {"A", "B", "F", "G"}, ["a", "cosi", "omega", "Omega"]
    )(A=2.0, B=0.0, F=0.0, G=1.0)

    assert elements == pytest.approx(
        {"a": 2.0, "cosi": 0.5, "omega": 0.0, "Omega": 0.0},
        abs=1e-12,
    )


def test_component_masses_and_semimajor_axes_follow_barycentric_relations():
    result = build_transform_functions(
        {"a", "M1", "M2"}, ["Mtot", "q", "a1", "a2"]
    )(a=12.0, M1=2.0, M2=4.0)

    assert result == pytest.approx(
        {"Mtot": 6.0, "q": 2.0, "a1": 8.0, "a2": 4.0}
    )


def test_total_mass_and_mass_ratio_recover_component_masses():
    result = build_transform_functions(
        {"Mtot", "q"}, ["M1", "M2"]
    )(Mtot=6.0, q=2.0)

    assert result == pytest.approx({"M1": 2.0, "M2": 4.0})


def test_radial_velocity_amplitude_and_projected_axis_are_consistent():
    values = {"a1": 3.0, "sini": 0.8, "n": 2.0, "e": 0.6}
    expected_k1 = 6.0

    k1 = transform(set(values), "K1", **values)
    a1sini = transform({"n", "K1", "e"}, "a1sini", n=2.0, K1=k1, e=0.6)

    assert k1 == pytest.approx(expected_k1)
    assert a1sini == pytest.approx(2.4)


def test_spectroscopic_mass_function_uses_eccentricity_squared():
    period = 2.0
    amplitude = 3.0
    eccentricity = 0.5
    expected = (
        period
        * amplitude**3
        * (1.0 - eccentricity**2) ** 1.5
        / (TWO_PI * GRAVITATIONAL_CONSTANT)
    )

    mass_function = transform(
        {"P", "K1", "e"},
        "f1",
        P=period,
        K1=amplitude,
        e=eccentricity,
    )

    assert mass_function == pytest.approx(expected)


def test_mass_function_from_masses_matches_projected_companion_mass():
    mass_function = transform(
        {"M2", "Mtot", "sini"},
        "f1",
        M2=1.0,
        Mtot=3.0,
        sini=0.5,
    )
    projected_mass = transform(
        {"f1", "Mtot"},
        "M2sini",
        f1=mass_function,
        Mtot=3.0,
    )

    assert mass_function == pytest.approx(1.0 / 72.0)
    assert projected_mass == pytest.approx(0.5)


def test_epoch_and_periastron_time_parameterizations_round_trip():
    periastron_time = transform(
        {"Tepoch", "P", "t0"},
        "Tp",
        Tepoch=10.0,
        P=4.0,
        t0=0.25,
    )
    scaled_time = transform(
        {"Tp", "Tepoch", "P"},
        "t0",
        Tp=periastron_time,
        Tepoch=10.0,
        P=4.0,
    )

    assert periastron_time == pytest.approx(11.0)
    assert scaled_time == pytest.approx(0.25)


def test_mean_anomaly_relates_epoch_periastron_and_mean_motion():
    mean_anomaly = transform(
        {"Tepoch", "n", "Tp"},
        "M0",
        Tepoch=12.0,
        n=0.5,
        Tp=10.0,
    )
    periastron_time = transform(
        {"Tepoch", "M0", "n"},
        "Tp",
        Tepoch=12.0,
        M0=mean_anomaly,
        n=0.5,
    )

    assert mean_anomaly == pytest.approx(1.0)
    assert periastron_time == pytest.approx(10.0)


def test_eccentric_anomaly_satisfies_keplers_equation():
    eccentricity = 0.4
    mean_anomaly = 1.1

    eccentric_anomaly = transform(
        {"e", "M0"},
        "E0",
        e=eccentricity,
        M0=mean_anomaly,
    )

    assert eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) == (
        pytest.approx(mean_anomaly)
    )


def test_eccentric_anomaly_recovers_mean_and_true_anomalies():
    eccentric_anomaly = 1.2
    eccentricity = 0.3
    expected_mean = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
    expected_true = np.arctan2(
        np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly),
        np.cos(eccentric_anomaly) - eccentricity,
    )

    result = build_transform_functions(
        {"E0", "e"}, ["M0", "nu0"]
    )(E0=eccentric_anomaly, e=eccentricity)

    assert result == pytest.approx({"M0": expected_mean, "nu0": expected_true})


def test_parallax_and_distance_are_reciprocals():
    distance = transform({"parallax"}, "distance", parallax=0.025)
    parallax = transform({"distance"}, "parallax", distance=distance)

    assert distance == pytest.approx(40.0)
    assert parallax == pytest.approx(0.025)


def test_overconstrained_parameters_finds_a_redundant_inverse_pair():
    assert overconstrained_parameters({"P", "n", "parallax"}) == {"P", "n"}


def test_overconstrained_parameters_finds_a_redundant_mass_triplet():
    assert overconstrained_parameters({"Mtot", "M1", "M2"}) == {
        "Mtot",
        "M1",
        "M2",
    }


def test_overconstrained_parameters_is_empty_for_independent_inputs():
    assert overconstrained_parameters({"P", "e", "parallax"}) == set()
