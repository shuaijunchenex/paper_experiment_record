"""Frozen verbatim function from ../broadcast_reconstruction/plot_figure.py.
See source_audit.md and provenance.json. No data-generation code imported.
"""
import numpy as np

def compact_sp_plus(
    b_global: np.ndarray, a_global: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[int, ...]]]:
    """Canonicalize LoRA factors using two thin QRs and an R x R core SVD."""
    if b_global.ndim != 2 or a_global.ndim != 2:
        raise ValueError("LoRA factors must be matrices")
    if b_global.shape[1] != a_global.shape[0]:
        raise ValueError("incompatible LoRA factor ranks")

    q_b, r_b = np.linalg.qr(b_global, mode="reduced")
    q_a, r_a = np.linalg.qr(a_global.T, mode="reduced")
    core = r_b @ r_a.T
    u_core, singular_values, vh_core = np.linalg.svd(core, full_matrices=False)
    sqrt_s = np.sqrt(np.maximum(singular_values, 0.0))
    b_plus = (q_b @ u_core) * sqrt_s[None, :]
    a_plus = sqrt_s[:, None] * (vh_core @ q_a.T)

    intermediate_shapes = {
        "q_b": q_b.shape,
        "r_b": r_b.shape,
        "q_a": q_a.shape,
        "r_a": r_a.shape,
        "core": core.shape,
        "u_core": u_core.shape,
        "vh_core": vh_core.shape,
        "b_plus": b_plus.shape,
        "a_plus": a_plus.shape,
    }
    dense_shape = (b_global.shape[0], a_global.shape[1])
    if any(shape == dense_shape for shape in intermediate_shapes.values()):
        raise AssertionError(f"compact path created a dense-shaped intermediate: {dense_shape}")
    return b_plus, a_plus, singular_values, intermediate_shapes
