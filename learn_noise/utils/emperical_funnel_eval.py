import numpy as np
import torch
import math
from numpy.polynomial.hermite import hermgauss

# ---------- Analytic funnel x2 marginal: pdf, cdf, moments ----------

def _analytic_funnel_x2_pdf(x2_grid: np.ndarray, scale1: float, gh_n: int = 80) -> np.ndarray:
    """
    x1 ~ N(0, scale1^2), x2 | x1 ~ N(0, exp(x1)).
    p(x2) = E_{x1}[ N(x2; 0, exp(x1)) ]   (lognormal variance mixture)
    """
    x2 = np.asarray(x2_grid, dtype=np.float64)
    nodes, weights = hermgauss(gh_n)
    x1_vals = (np.sqrt(2.0) * scale1) * nodes
    w_norm = weights / np.sqrt(np.pi)
    var = np.exp(x1_vals)                  # (N,)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    var_col = var[:, None]
    coef = inv_sqrt_2pi / np.sqrt(var_col)
    expo = np.exp(- (x2[None, :]**2) / (2.0 * var_col))
    pdf_matrix = coef * expo
    pdf = (w_norm[:, None] * pdf_matrix).sum(axis=0)
    return np.maximum(pdf, 1e-300)

def _analytic_funnel_x2_cdf(x2_grid: np.ndarray, scale1: float, gh_n: int = 200) -> np.ndarray:
    """
    F(x) = E_{x1~N(0,scale1^2)}[ Φ( x / sqrt(exp(x1)) ) ]  via Gauss–Hermite,
    computed in torch so we can use torch.erf.
    """
    # inputs -> torch (float64 for stability)
    x2 = torch.as_tensor(x2_grid, dtype=torch.float64)

    nodes_np, weights_np = hermgauss(gh_n)
    nodes   = torch.as_tensor(nodes_np,   dtype=torch.float64)
    weights = torch.as_tensor(weights_np, dtype=torch.float64)

    # transform nodes to N(0, scale1^2)
    x1_vals = (math.sqrt(2.0) * float(scale1)) * nodes              # (N,)
    w_norm  = weights / math.sqrt(math.pi)                          # (N,)

    var = torch.exp(x1_vals)                                        # (N,)
    z = x2.unsqueeze(0) / torch.sqrt(var).unsqueeze(1)              # (N, M)

    cdf_matrix = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))        # (N, M)
    cdf = (w_norm.unsqueeze(1) * cdf_matrix).sum(dim=0)             # (M,)

    return torch.clamp(cdf, 0.0, 1.0).cpu().numpy()

def _ppf_from_cdf_analytic(p: float, scale1: float, gh_n: int = 200,
                           lo_init: float = -5_000.0, hi_init: float = 5_000.0,
                           tol: float = 1e-8, max_iter: int = 200) -> float:
    """
    Invert the analytic CDF with bisection.
    """
    assert 0.0 < p < 1.0
    lo, hi = float(lo_init), float(hi_init)
    # expand bracket if needed
    for _ in range(200):
        Flo, Fhi = _analytic_funnel_x2_cdf(np.array([lo]), scale1, gh_n)[0], \
                   _analytic_funnel_x2_cdf(np.array([hi]), scale1, gh_n)[0]
        if Flo <= p <= Fhi:
            break
        # expand symmetrically
        lo *= 2.0
        hi *= 2.0
    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        Fmid = _analytic_funnel_x2_cdf(np.array([mid]), scale1, gh_n)[0]
        if abs(hi - lo) < tol:
            return mid
        if Fmid < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def _true_kurtosis_x2(scale1: float) -> float:
    """
    Kurtosis (Pearson, not excess):
      E[x2^2] = exp(0.5*scale1^2)
      E[x2^4] = 3 * exp(2*scale1^2)
      K = E[x^4]/(E[x^2]^2) = 3 * exp(scale1^2)
    """
    return float(3.0 * np.exp(scale1**2))

# ---------- Sample stats (skewness/kurtosis in the usual definitions) ----------

def _sample_skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0.0:
        return 0.0
    return float(np.mean(((x - m) / s)**3))

def _sample_kurtosis(x: np.ndarray) -> float:
    """
    Pearson kurtosis: E[((x-m)/s)^4] (normal -> 3).
    """
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    v = x.var(ddof=0)
    if v == 0.0:
        return 3.0
    return float(np.mean(((x - m)**4)) / (v**2))

# ---------- KS@tails (one-sample KS against analytic CDF, restricted to tails) ----------

def _ks1_tail_against_analytic(x: np.ndarray,
                               scale1: float,
                               q: float = 1e-3,
                               side: str = "right",
                               gh_n: int = 200) -> float:
    """
    One-sample KS statistic on a tail region vs analytic CDF.
    side ∈ {"right","left"}; q is the tail mass (e.g., 0.001).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    # full-sample sort for ECDF
    xs = np.sort(x)
    if side == "right":
        thr = _ppf_from_cdf_analytic(1.0 - q, scale1, gh_n)
        mask = xs >= thr
    else:
        thr = _ppf_from_cdf_analytic(q, scale1, gh_n)
        mask = xs <= thr

    x_tail = xs[mask]
    if x_tail.size == 0:
        return 0.0

    # Empirical CDF on the whole distribution, evaluated at tail points
    # (fraction of ALL samples ≤ x).
    # Because x_tail ⊆ xs and xs is sorted, ranks can be found via searchsorted:
    ranks = np.searchsorted(xs, x_tail, side="right")
    F_emp = ranks / float(n)

    # Analytic CDF at the same points
    F_true = _analytic_funnel_x2_cdf(x_tail, scale1, gh_n)

    return float(np.max(np.abs(F_emp - F_true)))

def ks_tails_analytic_x2(x: np.ndarray,
                         scale1: float,
                         q: float = 1e-3,
                         gh_n: int = 80):
    ks_right = _ks1_tail_against_analytic(x, scale1, q=q, side="right", gh_n=gh_n)
    ks_left  = _ks1_tail_against_analytic(x, scale1, q=q, side="left",  gh_n=gh_n)
    return ks_left, ks_right, 0.5 * (ks_left + ks_right)


# ---------- Public: evaluate x2-only metrics and (optionally) log to W&B ----------

def evaluate_x2_marginal_metrics(generated: np.ndarray,
                                 scale1: float = 3,
                                 tail_q: float = 1e-3,
                                 gh_n: int = 200,
                                 log_to_wandb: bool = False,
                                 step: int = None):
    """
    Returns dict with KR (vs analytic x2 kurtosis), |skewness|, and KS@tails vs analytic CDF.
    Only uses the non-Gaussian marginal x2.
    """
    # accept torch tensor
    if torch.is_tensor(generated):
        gen = generated.detach().cpu().numpy()
    else:
        gen = np.asarray(generated)

    x2 = gen[:, 1].astype(np.float64)

    # KR vs analytic kurtosis
    k_true = _true_kurtosis_x2(scale1)
    k_sim  = _sample_kurtosis(x2)
    KR = abs(1.0 - (k_sim / k_true))

    # "SR": true skewness = 0, so we report |skewness_sim|
    skew_sim = abs(_sample_skewness(x2))

    # KS on tails vs analytic CDF
    ks_left, ks_right, ks_avg = ks_tails_analytic_x2(x2, scale1, q=tail_q, gh_n=gh_n)


    out = {
        "x2/KR": KR,
        "x2/|skewness|": skew_sim,
        "x2/KS": ks_avg ,
    }
    # logging disabled (handled upstream)

    return out


# ============================================================================
# Two-sample x2-marginal evaluations (unnormalized & normalized)
# ----------------------------------------------------------------------------
# Given two empirical distributions A and B (arrays of shape [N,2] and [M,2]
# holding samples of (x1, x2)), we compare only the *non-Gaussian marginal* x2
# in the same spirit as the one-sample metrics above.
#
# We report, for both RAW (unnormalized) space and Z-SCORED (normalized) space:
#   - Two-sample KS on left/right tails and their average, D in [0,1]
#   - The size-normalized KS statistic  sqrt(n*m/(n+m)) * D  (can exceed 1)
#   - Absolute difference in Pearson kurtosis  |K_A - K_B|
#   - Absolute difference in skewness          |S_A - S_B|
#
# Tail region is defined *symmetrically* via pooled quantiles:
#   left  tail: x <= Q_pool(q)
#   right tail: x >= Q_pool(1-q)
# ============================================================================

import numpy as _np

def _zscore_np(x: _np.ndarray, eps: float = 1e-12):
    x = _np.asarray(x, dtype=_np.float64)
    m = x.mean()
    s = x.std(ddof=0)
    if not _np.isfinite(s) or s < eps:
        s = 1.0
    return (x - m) / s, float(m), float(s)

def _two_sample_ecdf_on_grid(xs_sorted: _np.ndarray, grid: _np.ndarray) -> _np.ndarray:
    """
    ECDF F(x) evaluated on 'grid' using a pre-sorted sample xs_sorted.
    Returns values in [0,1].
    """
    n = xs_sorted.size
    if n == 0:
        return _np.zeros_like(grid, dtype=_np.float64)
    ranks = _np.searchsorted(xs_sorted, grid, side="right")
    return ranks / float(n)

def _two_sample_ks_tail(xa: _np.ndarray,
                        xb: _np.ndarray,
                        q: float = 1e-3,
                        side: str = "right"):
    """
    Two-sample KS on a tail region determined by pooled quantiles.
    Returns (D, D_norm) where D ∈ [0,1] and
      D_norm = sqrt(n*m/(n+m)) * D  (size-normalized KS).
    """
    xa = _np.asarray(xa, dtype=_np.float64)
    xb = _np.asarray(xb, dtype=_np.float64)
    na, nb = xa.size, xb.size
    if na == 0 or nb == 0:
        return 0.0, 0.0

    # Sort full samples for ECDFs
    sa = _np.sort(xa)
    sb = _np.sort(xb)

    # Tail thresholds from pooled sample
    pool = _np.concatenate([sa, sb], axis=0)
    if side == "right":
        thr = _np.quantile(pool, 1.0 - q)
        ta = sa[sa >= thr]; tb = sb[sb >= thr]
    else:
        thr = _np.quantile(pool, q)
        ta = sa[sa <= thr]; tb = sb[sb <= thr]

    if ta.size == 0 and tb.size == 0:
        return 0.0, 0.0

    # Evaluate on the union grid of both tail samples
    grid = _np.sort(_np.unique(_np.concatenate([ta, tb])))
    Fa = _two_sample_ecdf_on_grid(sa, grid)
    Fb = _two_sample_ecdf_on_grid(sb, grid)

    D = float(_np.max(_np.abs(Fa - Fb)))
    # Size-normalized KS (useful for significance; can exceed 1)
    factor = _np.sqrt((na * nb) / float(na + nb))
    D_norm = float(factor * D)
    return D, D_norm

def ks_tails_two_sample(x2_A: _np.ndarray,
                        x2_B: _np.ndarray,
                        q: float = 1e-3):
    """
    Convenience wrapper: compute two-sample KS on left/right tails.
    Returns dict with raw and size-normalized scores.
    """
    Dl, Dl_norm = _two_sample_ks_tail(x2_A, x2_B, q=q, side="left")
    Dr, Dr_norm = _two_sample_ks_tail(x2_A, x2_B, q=q, side="right")
    out = {
        "KS_left":  Dl,
        "KS_right": Dr,
        "KS":       0.5 * (Dl + Dr),
        "KS_left_norm":  Dl_norm,
        "KS_right_norm": Dr_norm,
        "KS_norm":       0.5 * (Dl_norm + Dr_norm),
    }
    return out

def evaluate_x2_two_sample_metrics(A: _np.ndarray,
                                   B: _np.ndarray,
                                   tail_q: float = 1e-3,
                                   log_to_wandb: bool = False,
                                   step: int = None):
    """
    Two-sample evaluation of the non-Gaussian marginal x2 between
    empirical distributions A and B (each either torch.Tensor or np.ndarray
    with shape [N,2] and [M,2]).
    Reports metrics in RAW space and in Z-SCORED space.
    """
    # Accept torch tensors
    if torch.is_tensor(A): A = A.detach().cpu().numpy()
    if torch.is_tensor(B): B = B.detach().cpu().numpy()

    A = _np.asarray(A); B = _np.asarray(B)
    x2A = A[:, 1].astype(_np.float64)
    x2B = B[:, 1].astype(_np.float64)

    # ---------- RAW (unnormalized) ----------
    ks_raw = ks_tails_two_sample(x2A, x2B, q=tail_q)

    kA_raw = _sample_kurtosis(x2A)
    kB_raw = _sample_kurtosis(x2B)
    sA_raw = _sample_skewness(x2A)
    sB_raw = _sample_skewness(x2B)

    # ---------- Z-SCORED (normalized) ----------
    zA, mA, sA = _zscore_np(x2A)
    zB, mB, sB = _zscore_np(x2B)
    ks_z = ks_tails_two_sample(zA, zB, q=tail_q)

    kA_z = _sample_kurtosis(zA)
    kB_z = _sample_kurtosis(zB)
    sA_z = _sample_skewness(zA)
    sB_z = _sample_skewness(zB)

    out = {
        # RAW
        "x2/two_sample/raw/KS":       ks_raw["KS"],
        "x2/two_sample/raw/KS_norm":       ks_raw["KS_norm"],
        "x2/two_sample/raw/Δkurtosis":  abs(1 - kA_raw / kB_raw),
        "x2/two_sample/raw/Δ|skewness|": abs(1 - sA_raw / sB_raw),

    }

    # logging disabled (handled upstream)

    return out
