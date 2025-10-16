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

# ---------- NEW: MACD on tails with ANALYTIC cutoffs ----------

def _macd_tail_analytic_cutoff_points(x: np.ndarray,
                                      scale1: float,
                                      q: float,
                                      side: str,
                                      gh_n: int = 200) -> float:
    """
    Mean Absolute CDF Deviation on the tail, using ANALYTIC cutoff, evaluated at SAMPLE tail points.
    Discrete estimator:
        MACD_tail ≈ (1 / (n q)) * sum_{x_i in tail} |F_n(x_i) - F(x_i)|
    where the tail set is {x_i : x_i ≥ F^{-1}(1-q)} for right (or ≤ F^{-1}(q) for left).
    """
    xs = np.sort(np.asarray(x, np.float64))
    n  = xs.size
    if n == 0:
        return 0.0

    if side == "right":
        thr = _ppf_from_cdf_analytic(1.0 - q, scale1, gh_n)
        tail = xs[xs >= thr]
    else:
        thr = _ppf_from_cdf_analytic(q, scale1, gh_n)
        tail = xs[xs <= thr]

    N_tail = tail.size
    if N_tail == 0:
        return 0.0

    # Empirical CDF at those points
    ranks = np.searchsorted(xs, tail, side="right")
    F_emp = ranks.astype(np.float64) / float(n)

    # True CDF at those points
    F_true = _analytic_funnel_x2_cdf(tail, scale1, gh_n)

    # Average absolute deviation over the tail, normalized by tail mass q
    macd = float(np.sum(np.abs(F_emp - F_true)) / (float(n) * float(q)))
    return macd


def _macd_tail_analytic_cutoff_grid(x: np.ndarray,
                                    scale1: float,
                                    q: float,
                                    side: str,
                                    M: int = 2048,
                                    gh_n: int = 200) -> float:
    """
    Mean Absolute CDF Deviation on the tail with ANALYTIC cutoff, evaluated on a fixed u-grid.
    This estimates:
        (1/q) * ∫_{tail} |F_n(x) - F(x)| dF(x)
      = (1/q) * ∫_{u in tail} |F_n(F^{-1}(u)) - u| du
    via midpoint quadrature in u-space (M points).
    """
    # Prepare data ECDF
    xs = np.sort(np.asarray(x, np.float64))
    n  = xs.size
    if n == 0:
        return 0.0

    if side == "right":
        # u-grid in [1-q, 1]
        us = np.linspace(1.0 - q, 1.0, M, endpoint=False) + 0.5 * q / M
    else:
        # u-grid in [0, q]
        us = np.linspace(0.0, q, M, endpoint=False) + 0.5 * q / M

    # Map u -> x by true PPF, then evaluate F_n(x)
    xs_grid = np.array([_ppf_from_cdf_analytic(float(u), scale1, gh_n) for u in us], dtype=np.float64)
    ranks   = np.searchsorted(xs, xs_grid, side="right")
    F_emp   = ranks.astype(np.float64) / float(n)

    # Average |F_n(x) - u| over the tail, renormalized by q
    macd = float(np.mean(np.abs(F_emp - us)) / q)
    return macd


def tails_analytic_cutoff_macd(x: np.ndarray,
                               scale1: float,
                               q: float = 1e-3,
                               gh_n: int = 200,
                               M_grid: int = 2048):
    """
    Compute MACD on both tails using ANALYTIC cutoffs, returning
    - point-based estimator (sample tail points)
    - grid-based estimator (u-grid)
    """
    macd_L_pts = _macd_tail_analytic_cutoff_points(x, scale1, q, side="left",  gh_n=gh_n)
    macd_R_pts = _macd_tail_analytic_cutoff_points(x, scale1, q, side="right", gh_n=gh_n)
    macd_pts_avg = 0.5 * (macd_L_pts + macd_R_pts)

    macd_L_grid = _macd_tail_analytic_cutoff_grid(x, scale1, q, side="left",  M=M_grid, gh_n=gh_n)
    macd_R_grid = _macd_tail_analytic_cutoff_grid(x, scale1, q, side="right", M=M_grid, gh_n=gh_n)
    macd_grid_avg = 0.5 * (macd_L_grid + macd_R_grid)

    return (
        (macd_L_pts,  macd_R_pts,  macd_pts_avg),
        (macd_L_grid, macd_R_grid, macd_grid_avg),
    )


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

    macd_L_pts = _macd_tail_analytic_cutoff_points(x2, scale1, tail_q, side="left",  gh_n=gh_n)
    macd_R_pts = _macd_tail_analytic_cutoff_points(x2, scale1, tail_q, side="right", gh_n=gh_n)
    macd_pts_avg = 0.5 * (macd_L_pts + macd_R_pts)

    out = {
        "x2/KR": KR,
        "x2/|skewness|": skew_sim,
        'x2/L1_tail': macd_pts_avg,
        "x2/KS": ks_avg ,
    }
    # logging disabled (handled upstream)

    return out
