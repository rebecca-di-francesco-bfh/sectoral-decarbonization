import numpy as np
from sklearn.covariance import LedoitWolf
import cvxpy as cp

def nearest_psd(A):
    """
    Convert a symmetric matrix to the nearest positive semi-definite (PSD) matrix.

    This is necessary because numerical errors can sometimes produce covariance matrices
    with slightly negative eigenvalues, which violates the PSD requirement for covariance.

    Method:
    1. Compute eigendecomposition of matrix A
    2. Set any negative eigenvalues to zero
    3. Reconstruct the matrix using the corrected eigenvalues

    Args:
        A: Square symmetric matrix (typically a covariance matrix)

    Returns:
        Nearest PSD matrix to A
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < 0] = 0  # Replace negative eigenvalues with zero
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def sigma_raw_fn(R_clean):
    """
    Calculate the raw (sample) covariance matrix from returns data.

    Args:
        R_clean: DataFrame of clean returns data (no NaN values)

    Returns:
        Sample covariance matrix
    """
    return R_clean.cov()

def sigma_shrink_fn(R_clean):
    """
    Calculate a shrunk covariance matrix using Ledoit-Wolf shrinkage estimator.

    Ledoit-Wolf shrinkage improves covariance estimation by shrinking the sample
    covariance towards a structured estimator (identity matrix). This is particularly
    useful when the number of observations is small relative to the number of assets.

    Benefits:
    - More stable estimates (especially with limited data)
    - Better conditioning (avoids numerical issues)
    - Improved out-of-sample performance

    Args:
        R_clean: DataFrame of clean returns data (no NaN values)

    Returns:
        tuple: (Regularized PSD covariance matrix, shrinkage intensity alpha)
    """
    lw = LedoitWolf().fit(R_clean)
    Sigma_shrink = lw.covariance_

    # Add small regularization term to diagonal for numerical stability
    Sigma_reg = Sigma_shrink + 1e-5 * np.eye(Sigma_shrink.shape[0])

    # Ensure PSD property
    return nearest_psd(Sigma_reg), lw.shrinkage_

def extract_optimal_portfolios_at_target_te(optimal_portfolios_all_te, target_te_bps=200):
    """
    Extracts the portfolio weights and carbon reduction levels closest to the target TE (bps)
    from the full carbon-TE frontier results.
    """
    optimal_portfolios_target_TE = {}

    for sector, info in optimal_portfolios_all_te.items():
        te_list = np.array(info["tracking_errors"])
        weights_list = info["weights_by_te"]

        if len(te_list) == 0:
            continue

        # find index of TE closest to target
        idx_closest = np.argmin(np.abs(te_list - target_te_bps))
        te_closest = te_list[idx_closest]
        
        w_opt = weights_list[idx_closest]

        optimal_portfolios_target_TE[sector] = {
            "cov_type": info["cov_type"],
            "diagnostics": info["diagnostics"],
            "w_opt": w_opt,
            "w_bench": info.get("w_bench"),
            "stock_labels": info.get("stock_labels"),
            "tracking_error_at_2pct": te_closest,
"carbon_reduction_at_2pct": info["carbon_reductions"][idx_closest]
        }

    return optimal_portfolios_target_TE

def solve_qp_with_fallback(prob):
    # 1. Try MOSEK strict
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate"):
            return
    except:
        pass

    # 2. Try MOSEK relaxed
    try:
        prob.solve(
        solver=cp.MOSEK,
        verbose=False,
        **{"mosek_params": {"MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-4}}
        )

        if prob.status in ("optimal", "optimal_inaccurate"):
            return
    except:
        pass
