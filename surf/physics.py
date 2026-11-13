"""Linear wave theory and the breaking-height formula. SI units throughout."""
import numpy as np

G = 9.81


def wavenumber(T, h):
    """Wavenumber k (rad/m) for period T (s) in depth h (m).

    Solves the linear dispersion relation  w^2 = g k tanh(k h)  by Newton's
    method, starting from the deep-water value k0 = w^2 / g.
    """
    w2 = (2 * np.pi / T) ** 2
    k = w2 / G
    for _ in range(30):
        t = np.tanh(k * h)
        k -= (G * k * t - w2) / (G * t + G * k * h * (1 - t * t))
    return k


def group_velocity(T, h):
    """Group velocity cg (m/s):  cg = (w/k) * 0.5 * (1 + 2kh / sinh(2kh))."""
    k = wavenumber(T, h)
    return (2 * np.pi / T) / k * 0.5 * (1 + 2 * k * h / np.sinh(2 * k * h))


def deep_water_height(H, T, h):
    """Undo shoaling: height H (m) at depth h -> equivalent deep-water height.

    H0 = H / Ks,  Ks = sqrt(cg0 / cg),  deep-water cg0 = g T / (4 pi).
    """
    return H / np.sqrt(G * T / (4 * np.pi) / group_velocity(T, h))


def breaking_height(H0, T):
    """Breaker height (m) from deep-water height H0 (m) and period T (s).

    Komar and Gaughan (1973):  Hb = 0.39 g^(1/5) (T H0^2)^(2/5).
    """
    return 0.39 * G ** 0.2 * (T * H0 ** 2) ** 0.4
