"""Each expected number below was worked out by hand (see comments)."""
import numpy as np
import pytest

from surf import physics


def test_wavenumber_satisfies_dispersion():
    for T, h in [(6, 5), (10, 10), (14, 15), (20, 10), (8, 4000)]:
        k = physics.wavenumber(T, h)
        assert physics.G * k * np.tanh(k * h) == pytest.approx((2 * np.pi / T) ** 2)


def test_wavenumber_hand_value():
    # T=10 s, h=10 m: w^2 h/g = 0.4024, so kh tanh(kh) = 0.4024 -> kh = 0.680
    assert physics.wavenumber(10, 10) * 10 == pytest.approx(0.680, abs=0.002)


def test_wavenumber_deep_water_limit():
    # deep water: k = w^2/g, wavelength = g T^2 / 2pi = 156.1 m for T=10 s
    assert 2 * np.pi / physics.wavenumber(10, 4000) == pytest.approx(156.1, abs=0.1)


def test_group_velocity_hand_value():
    # T=10 s, h=10 m: c = w/k = 9.24 m/s, n = 0.5(1 + 1.36/sinh 1.36) = 0.874
    assert physics.group_velocity(10, 10) == pytest.approx(8.07, abs=0.03)


def test_group_velocity_limits():
    # deep: cg = gT/4pi = 7.81 m/s.  shallow: cg = sqrt(g h) = 3.13 m/s at h=1 m
    assert physics.group_velocity(10, 4000) == pytest.approx(7.81, abs=0.01)
    assert physics.group_velocity(20, 1) == pytest.approx(3.13, abs=0.02)


def test_deep_water_height_hand_value():
    # T=14 s, h=15 m: cg = 10.38, cg0 = 10.93, Ks = 1.026 -> H0 = 1/1.026
    assert physics.deep_water_height(1.0, 14, 15) == pytest.approx(0.975, abs=0.003)
    assert physics.deep_water_height(1.0, 10, 4000) == pytest.approx(1.0, abs=1e-6)


def test_breaking_height_hand_value():
    # H0=2 m, T=14 s: 0.39 * 9.81^0.2 * (14*4)^0.4 = 0.39 * 1.579 * 5.00 = 3.08
    assert physics.breaking_height(2.0, 14) == pytest.approx(3.08, abs=0.01)


def test_longer_period_breaks_bigger():
    assert physics.breaking_height(1.0, 16) > 1.25 * physics.breaking_height(1.0, 8)
