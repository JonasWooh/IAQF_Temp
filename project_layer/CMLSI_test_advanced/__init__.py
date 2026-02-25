"""
CMLSI Advanced Robustness Tests
Step 1: Prove structure changed (covariance, PCA subspace)
Step 2: Stable vs Adaptive CMLSI
Step 3: Compare in VAR (IRF/FEVD)
Step 4: Economic interpretation
"""
from .structure_tests import run_structure_tests
from .rolling_pca import compute_adaptive_cmlsi, run_adaptive_cmlsi
from .var_comparison import run_stable_vs_adaptive_comparison
from .runner import run_full_cmlsi_robustness

__all__ = [
    "run_structure_tests",
    "compute_adaptive_cmlsi",
    "run_adaptive_cmlsi",
    "run_stable_vs_adaptive_comparison",
    "run_full_cmlsi_robustness",
]
