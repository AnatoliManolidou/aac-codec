import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.linalg import solve_toeplitz
import os

# Path to the band table file
TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'TableB219.mat')

# TNS linear prediction order
P = 4

def tns(frame_F_in, frame_type):
    """
    Temporal Noise Shaping analysis for a single channel.
    Computes LP coefficients from normalized MDCT coefficients and applies
    the FIR filter H_TNS(z) = 1 - a1*z^-1 - ... - ap*z^-p to the original coefficients.

    Args:
        frame_F_in: MDCT coefficients before TNS
            - For 'ESH': shape (128, 8)
            - For 'OLS', 'LSS', 'LPS': shape (1024, 1)
        frame_type: Frame type ('OLS', 'LSS', 'ESH', 'LPS')

    Returns:
        frame_F_out: MDCT coefficients after TNS, same shape as frame_F_in
        tns_coeffs: Quantized TNS LP coefficients
            - For 'ESH': shape (4, 8)
            - For 'OLS', 'LSS', 'LPS': shape (4, 1)
    """

    table = loadmat(TABLE_PATH)

    if frame_type == 'ESH':
        band_table = table['B219b']  # 42 bands for short frames
        num_coeffs = 128
        num_subframes = 8
    else:
        band_table = table['B219a']  # 69 bands for long frames
        num_coeffs = 1024
        num_subframes = 1

    num_bands = band_table.shape[0]
    tns_coeffs = np.zeros((P, num_subframes))
    frame_F_out = frame_F_in.copy()

    # ===== PROCESS EACH SUBFRAME =====
    for sub in range(num_subframes):
        if frame_type == 'ESH':
            X = frame_F_in[:, sub].copy() # Processing each subframe
        else:
            X = frame_F_in[:, 0].copy()

        # ===== COMPUTE BAND ENERGIES =====
        S_w = np.zeros(num_coeffs)
        for j in range(num_bands):
            w_low = int(band_table[j, 1])
            w_high = int(band_table[j, 2])
            energy = np.sum(X[w_low:w_high + 1] ** 2)
            S_w[w_low:w_high + 1] = np.sqrt(energy)

        # ===== SMOOTH S_w (TWO-PASS AVERAGING) =====
        # Pass 1: backward
        for k in range(num_coeffs - 2, -1, -1):
            S_w[k] = (S_w[k] + S_w[k + 1]) / 2.0

        # Pass 2: forward
        for k in range(1, num_coeffs):
            S_w[k] = (S_w[k] + S_w[k - 1]) / 2.0

        # ===== NORMALIZE MDCT COEFFICIENTS =====
        X_w = np.zeros(num_coeffs)
        for k in range(num_coeffs):
            if S_w[k] > 0:
                X_w[k] = X[k] / S_w[k]
            else:
                X_w[k] = 0.0

        # ===== COMPUTE AUTOCORRELATION =====
        r = np.zeros(P + 1)
        for lag in range(P + 1):
            r[lag] = np.sum(X_w[lag:] * X_w[:num_coeffs - lag])

        # ===== SOLVE NORMAL EQUATIONS (Ra = r) =====
        if r[0] < 1e-10:
            a = np.zeros(P)
        else:
            R_col = r[:P]  # First column of Toeplitz matrix 
            r_vec = r[1:P + 1]  # Right-hand side 
            try:
                a = solve_toeplitz(R_col, r_vec)
            except np.linalg.LinAlgError:
                a = np.zeros(P)

        # ===== QUANTIZE LP COEFFICIENTS (4-bit, step 0.1) =====
        a_quantized = np.round(a / 0.1) * 0.1
        a_quantized = np.clip(a_quantized, -0.8, 0.7)

        # ===== STABILITY CHECK =====
        poly = np.concatenate(([1.0], -a_quantized))
        roots = np.roots(poly)

        if np.any(np.abs(roots) >= 1.0):
            a_quantized = np.zeros(P)

        tns_coeffs[:, sub] = a_quantized

        # ===== APPLY FIR FILTER H_TNS(z) =====
        b_fir = np.concatenate(([1.0], -a_quantized))
        Y = lfilter(b_fir, [1.0], X)

        if frame_type == 'ESH':
            frame_F_out[:, sub] = Y
        else:
            frame_F_out[:, 0] = Y

    return frame_F_out, tns_coeffs


def i_tns(frame_F_in, frame_type, tns_coeffs):
    """
    Inverse Temporal Noise Shaping (TNS).
    Applies the inverse filter H_TNS^-1(z) to reconstruct the original MDCT coefficients.

    Args:
        frame_F_in: MDCT coefficients after TNS
            - For 'ESH': shape (128, 8)
            - For 'OLS', 'LSS', 'LPS': shape (1024, 1)
        frame_type: Frame type ('OLS', 'LSS', 'ESH', 'LPS')
        tns_coeffs: Quantized TNS LP coefficients
            - For 'ESH': shape (4, 8)
            - For 'OLS', 'LSS', 'LPS': shape (4, 1)

    Returns:
        frame_F_out: Reconstructed MDCT coefficients, same shape as frame_F_in
    """
    
    if frame_type == 'ESH':
        num_subframes = 8
    else:
        num_subframes = 1

    frame_F_out = frame_F_in.copy()

    # ===== PROCESS EACH SUBFRAME =====
    for sub in range(num_subframes):
        a_quantized = tns_coeffs[:, sub]

        if frame_type == 'ESH':
            Y = frame_F_in[:, sub].copy()
        else:
            Y = frame_F_in[:, 0].copy()

        # ===== APPLY INVERSE FILTER H_TNS^-1(z) =====
        a_iir = np.concatenate(([1.0], -a_quantized)) # Denominator coefficients for IIR filter
        X = lfilter([1.0], a_iir, Y)

        if frame_type == 'ESH':
            frame_F_out[:, sub] = X
        else:
            frame_F_out[:, 0] = X

    return frame_F_out