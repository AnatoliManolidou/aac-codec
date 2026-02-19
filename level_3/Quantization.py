import numpy as np
from scipy.io import loadmat
import os

# Path to the band table file
TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'TableB219.mat')

def aac_quantizer(frame_F, frame_type, SMR):

    """
    Calculates the threshold T(b) for each band and implements the quantization

    Args:
        frame_F: the current frame in frequency domain
        frame_type: the type of the current frame
        SMR: Signal to Mask Ration, dimentions (42, 8) for ESH frames and (69, 1) for the rest types of frames

    Returns:
        S: The quantized symbols of the MDCT coefficients of the current frame, a 1024x1 matrix
        sfc: A 42x1 vector for ESH frames and a 69x1 vector for the rest types of frames, containing the scalefactors of each band
        G: The global gain of the current frame, 1x8 vector for ESH frames and a scalar for the rest types of frames
    """

    table = loadmat(TABLE_PATH)

    if frame_type == 'ESH':
        band_table = table['B219b']  # 42 bands for short frames
        N = 256
        num_coeff = 128
        num_subframes = 8
        
    else:
        band_table = table['B219a']  # 69 bands for long frames
        N = 2048
        num_coeff = 1024
        num_subframes = 1

    MQ = 8191 
    magic_num = 0.4054

    S = np.zeros(frame_F.shape[0])
    sfc = np.zeros((SMR.shape[0], num_subframes))
    G = np.zeros(num_subframes)

    for k in range(num_subframes):

        offset = k * num_coeff

        X_sub = frame_F[offset : offset + num_coeff]

        # ===== CALCULATING THE THRESHOLD T(b) FOR EACH BAND =====
        P_b = np.zeros(SMR.shape[0])
        T_b = np.zeros(SMR.shape[0])

        for b in range(SMR.shape[0]):
            start_idx = int(band_table[b, 1])
            end_idx = int(band_table[b, 2]) + 1

            P_b[b] = np.sum(X_sub[start_idx:end_idx] ** 2)

            if SMR[b, k] > 0:
                T_b[b] = P_b[b] / SMR[b, k]
            else:
                T_b[b] = 0

        # ===== CALCULATING THE INITIAL SCALEFACTOR GAIN a =====

        max_mdct = max(np.max(np.abs(X_sub)), 1e-10)  # Avoid division by zero

        a_hat = int((16/3) * np.log2(max_mdct ** (3/4) / MQ))

        # ===== FINDING THE OPTIMAL a THOUGH ITERATION =====

        alpha = np.zeros(SMR.shape[0])

        for b in range(SMR.shape[0]):
            start_idx = int(band_table[b, 1])
            end_idx = int(band_table[b, 2]) + 1

            X_band = X_sub[start_idx:end_idx]

            a = a_hat

            while True:
                S_band = np.sign(X_band) * np.floor((np.abs(X_band) * (2**(-0.25*a)))** (3/4) + magic_num)

                # If all symbols are quantized to zero, no point increasing a further
                if np.all(S_band == 0):
                    break

                X_hat_band = np.sign(S_band) * (np.abs(S_band) ** (4/3)) * (2**(0.25*a))

                Pe_b = np.sum((X_band - X_hat_band) ** 2)

                if Pe_b < T_b[b]:
                    a += 1

                    if b > 0 and abs(a - alpha[b-1]) > 60:
                        break
                else:
                    break

            alpha[b] = a
            S[offset + start_idx : offset + end_idx] = S_band

        # ===== CALCULATE THE GLOBAL GAIN =====
        G[k] = alpha[0]

        # ===== CALCULATE THE SCALE FACTORS =====

        sfc[0, k] = alpha[0]
        for b in range(1, SMR.shape[0]):
            sfc[b, k] = alpha[b] - alpha[b -1]

    if frame_type != 'ESH':
        G = G[0]  # For long frames, G is a scalar

    return S, sfc, G

def i_aac_quantizer(S, sfc, G, frame_type):

    """
    Implements the inverse quantization
    
    Args:
        S: The quantized symbols of the MDCT coefficients of the current frame, a 1024x1 matrix
        sfc: A 42x1 vector for ESH frames and a 69x1 vector for the rest types of frames, containing the scalefactors of each band
        G: The global gain of the current frame, 1x8 vector for ESH frames and a scalar for the rest types of frames
        frame_type: the type of the current frame

    Returns:

        frame_F: the reconstructed frame in frequency domain, a 1024x1 matrix
    
    """
    table = loadmat(TABLE_PATH)

    if frame_type == 'ESH':
        band_table = table['B219b']  # 42 bands for short frames
        N = 256
        num_coeff = 128
        num_subframes = 8
        G_vec = G
        
    else:
        band_table = table['B219a']  # 69 bands for long frames
        N = 2048
        num_coeff = 1024
        num_subframes = 1
        G_vec = [G]

    frame_F = np.zeros(S.shape[0])

    for k in range(num_subframes):

        offset = k * num_coeff

        S_sub = S[offset : offset + num_coeff]

        # ===== INVERSE DPCM FOR SCALE FACTORS =====
        alpha = np.zeros(band_table.shape[0])

        alpha[0] = G_vec[k]

        for b in range(1, band_table.shape[0]):
            alpha[b] = alpha[b - 1] + sfc[b, k]

        # ===== INVERSE QUANTIZATION =====

        for b in range(band_table.shape[0]):
            start_idx = int(band_table[b, 1])
            end_idx = int(band_table[b, 2]) + 1

            S_band = S_sub[start_idx : end_idx]

            X_hat_band = np.sign(S_band) * (np.abs(S_band) ** (4/3)) * (2**(0.25*alpha[b]))

            frame_F[offset + start_idx : offset + end_idx] = X_hat_band

    return frame_F