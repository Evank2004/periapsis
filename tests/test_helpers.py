import pytest
import numpy as np
from periapsis.utils.helpers import error_sigfig_latex_string

def test_error_sigfig_latex_string_basic():
    assert error_sigfig_latex_string(1.2345, 0.1234, 0.1234) == r"1.23^{+0.12}_{-0.12}"
    assert error_sigfig_latex_string(1.2345, 0.001234, 0.001234) == r"1.2345^{+0.0012}_{-0.0012}"
    assert error_sigfig_latex_string(1.2345, 0.0001234, 0.0001234) == r"1.23450^{+0.00012}_{-0.00012}"
    assert error_sigfig_latex_string(1.2345, 0.4321, 0.3842) == r"1.2^{+0.4}_{-0.4}"


def test_error_sigfig_latex_string_with_asymmetric_errors():
    assert error_sigfig_latex_string(1.2345, 0.246, 0.123) == r"1.23^{+0.25}_{-0.12}"
    assert error_sigfig_latex_string(1.2345, 0.001234, 0.345) == r"1.2345^{+0.0012}_{-0.3450}"
    assert error_sigfig_latex_string(1.2345, 0.345, 0.001234) == r"1.2345^{+0.3450}_{-0.0012}"

def test_error_sigfig_latex_string_with_large_values():
    assert error_sigfig_latex_string(12345, 1234, 1234) == r"1.23^{+0.12}_{-0.12} \times 10^{4}"
    assert error_sigfig_latex_string(12345, 123, 321) == r"1.234^{+0.012}_{-0.032} \times 10^{4}"
    assert error_sigfig_latex_string(12345, 0.001234, 0.004321) == r"1.23450000^{+0.00000012}_{-0.00000043} \times 10^{4}"

def test_error_sigfig_latex_string_with_small_value_large_error():
    assert error_sigfig_latex_string(0.1, 124, 766) == r"0.0^{+1.2}_{-7.7} \times 10^{2}"

def test_pdg_950_boundary():
    # 0.0954 rounds to 0.10 (2 sig figs) NOT 0.1 (1 sig fig)
    assert error_sigfig_latex_string(1.234, 0.0954, 0.0954) == r"1.23^{+0.10}_{-0.10}"
    # 0.962 rounds to 1.0
    assert error_sigfig_latex_string(12.34, 0.962, 0.962) == r"12.3^{+1.0}_{-1.0}"
    assert error_sigfig_latex_string(1.234, 0.0949, 0.0949) == r"1.23^{+0.09}_{-0.09}"

def test_pdg_exact_boundaries():
    # 354 -> 2 sig figs (0.35)
    assert error_sigfig_latex_string(1.234, 0.354, 0.354) == r"1.23^{+0.35}_{-0.35}"
    # 355 -> 1 sig fig (0.4)
    assert error_sigfig_latex_string(1.234, 0.355, 0.355) == r"1.2^{+0.4}_{-0.4}"
    # 949 -> 1 sig fig (0.9)
    assert error_sigfig_latex_string(1.234, 0.949, 0.949) == r"1.2^{+0.9}_{-0.9}"

def test_negative_central_values():
    assert error_sigfig_latex_string(-1.2345, 0.1234, 0.1234) == r"-1.23^{+0.12}_{-0.12}"
    assert error_sigfig_latex_string(-12345, 1234, 1234) == r"-1.23^{+0.12}_{-0.12} \times 10^{4}"
    # Negative value close to zero with large error
    assert error_sigfig_latex_string(-0.1, 124, 766) == r"0.0^{+1.2}_{-7.7} \times 10^{2}" # Note: avoid "-0.0"

def test_negative_zero_prevention():
    assert error_sigfig_latex_string(-0.001, 0.12, 0.12) == r"0.00^{+0.12}_{-0.12}"

def test_small_values_negative_exponents():
    assert error_sigfig_latex_string(1.2345e-5, 1.234e-6, 1.234e-6) == r"1.23^{+0.12}_{-0.12} \times 10^{-5}"
    assert error_sigfig_latex_string(1.2345e-5, 4.321e-4, 3.842e-4) == r"0.0^{+0.4}_{-0.4} \times 10^{-3}" # Or 0^{+4}_{-4} \times 10^{-4}

def test_central_value_rounding_rollover():
    assert error_sigfig_latex_string(99995, 120, 120) == r"1.000^{+0.012}_{-0.012} \times 10^{5}"

def test_intermediate_values():
    assert error_sigfig_latex_string(542, 36, 91) == r"540^{+40}_{-90}"

def test_halfway_rounding():
    # Locking in expected behavior for exact 5s
    assert error_sigfig_latex_string(1.235, 0.125, 0.135) == r"1.24^{+0.12}_{-0.14}" # (if round-half-to-even)

def test_invalid_or_degenerate_inputs():
    with pytest.raises(ValueError):
        error_sigfig_latex_string(1.23, -0.1, 0.1)  # Negative uncertainty
    with pytest.raises(ValueError):
        error_sigfig_latex_string(1.23, 0.0, 0.0)   # Zero uncertainty
    with pytest.raises(ValueError):
        error_sigfig_latex_string(np.nan, 0.1, 0.1) # NaNs