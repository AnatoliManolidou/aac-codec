import numpy as np
from scipy.io import loadmat
import os

# Path to the band table file
TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'TableB219.mat')

def spreading_function(i, j, band_table):

    """
    Computes the spreading function value for the given band indices i and j for long and short frames.
    
    Args:
        i: index of the band for which we are calculating the masking threshold
        j: index of the band that is potentially masking band i
        band_table: the band table for the current frame type

    Returns:
        x: Table value of the spreading function for the given band indices i and j
    """

    bval = band_table[:, 4]  # bval is in the 5th column of the band table

    tmpx = 0
    tmpz = 0
    tmpy = 0
    x = 0

    if i >= j:
        tmpx = 3 * (bval[j] - bval[i])
    else:
        tmpx = 1.5 * (bval[j] - bval[i])
    
    tmpz = (8 * min(((tmpx - 0.5)**2) - 2*(tmpx - 0.5), 0))
    tmpy = (15.811389 + 7.5 * (tmpx + 0.474) - 17.5 * (1 + (tmpx + 0.474)**2)**0.5)

    if tmpy < -100:
        x = 0
    else: 
        x = 10 ** ((tmpy + tmpz)/ 10)

    return x


def psycho(frame_T, frame_type, frame_T_prev_1, frame_T_prev_2):

    """

    Implements the phychoacoustic model for one channel 

    Args:
        frame_T: the current frame in time domain
        frame_type: the type of the current frame
        frame_T_prev_1: the previous frame of frame_T in the same channel 
        frame_T_prev_2: the previous previous frame of frame_T in the same channel

    Returns:
        SMR: Signal to Mask Ration, dimentions (42, 8) for ESH frames and (69, 1) for the rest types of frames
    """
    table = loadmat(TABLE_PATH)

    if frame_type == 'ESH':
        band_table = table['B219b']  # 42 bands for short frames
        N = 256
        num_subframes = 8
        
    else:
        band_table = table['B219a']  # 69 bands for long frames
        N = 2048
        num_subframes = 1

    # ===== SPREADING FUNCTION CALCULATION =====

    # Compute the spreading function values for all pairs of bands
    spreading_matrix = np.zeros((band_table.shape[0], band_table.shape[0]))
    for i in range(band_table.shape[0]):
        for j in range(band_table.shape[0]):
            spreading_matrix[i, j] = spreading_function(i, j, band_table)

    SMR = np.zeros((band_table.shape[0], num_subframes))

    for k in range(num_subframes):

        subframe_T = frame_T[k * N : (k + 1) * N]
        subframe_T_prev_1 = frame_T_prev_1[k * N : (k + 1) * N]
        subframe_T_prev_2 = frame_T_prev_2[k * N : (k + 1) * N]

        # ===== MULTIPLICTION WITH HANN WINDOW AND FFT =====

        # Apply Hann window to the current frame in time domain
        s_w = np.zeros(N)
        for n in range(N):
            s_w[n] = subframe_T[n] * (0.5 - 0.5 * np.cos((np.pi * (n + 0.5))/ N))

        # Apply Hann window to the previous frame in time domain
        s_w_prev_1 = np.zeros(N)
        for n in range(N):
            s_w_prev_1[n] = subframe_T_prev_1[n] * (0.5 - 0.5 * np.cos((np.pi * (n + 0.5))/ N))

        # Apply Hann window to the previous previous frame in time domain
        s_w_prev_2 = np.zeros(N)
        for n in range(N):
            s_w_prev_2[n] = subframe_T_prev_2[n] * (0.5 - 0.5 * np.cos((np.pi * (n + 0.5))/ N))

        # === FFT CALCULATION KEEPING ONLY THE FIRST N/2 COEFFICIENTS DUE TO SYMMETRY ===

        # FFT of the current frame, extracting magnitude and phase of the FFT coefficients
        S = np.fft.fft(s_w, n=N)
        r = np.abs(S[:N//2]) # Magnitude
        f = np.angle(S[:N//2]) # Phase

        # FFT of the previous frame, extracting magnitude and phase of the FFT coefficients
        S_prev_1 = np.fft.fft(s_w_prev_1, n=N)
        r_1 = np.abs(S_prev_1[:N//2]) # Magnitude
        f_1 = np.angle(S_prev_1[:N//2]) # Phase

        # FFT of the previous previous frame, extracting magnitude and phase of the FFT coefficients
        S_prev_2 = np.fft.fft(s_w_prev_2, n=N)
        r_2 = np.abs(S_prev_2[:N//2]) # Magnitude
        f_2 = np.angle(S_prev_2[:N//2]) # Phase

        # ===== CALCULATING EXPECTED VALUES OF MAGNITUDE AND PHASE =====

        r_pred = np.zeros(N//2)
        f_pred = np.zeros(N//2)

        for w in range(N//2):
            r_pred[w] = 2 * r_1[w] - r_2[w]
            f_pred[w] = 2 * f_1[w] - f_2[w]

        # ===== CALCULATING MAGNITUDE OF PREDICTABILITY =====

        c = np.zeros(N//2)

        for w in range(N//2):
            first_term = (r[w] * np.cos(f[w]) - r_pred[w] * np.cos(f_pred[w])) ** 2
            second_term = (r[w] * np.sin(f[w]) - r_pred[w] * np.sin(f_pred[w])) ** 2
            denom = r[w] + np.abs(r_pred[w])
            if denom > 0:
                c[w] = np.sqrt(first_term + second_term) / denom
            else:
                c[w] = 0  # Both current and predicted are zero

        # ===== CALCULATING THE ENERGY AND THE WEIGHTED PREDICTABILITY =====

        e_b = np.zeros(band_table.shape[0])
        c_b = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):

            start_idx = int(band_table[b, 1])
            end_idx = int(band_table[b, 2]) + 1

            e_b[b] = np.sum(r[start_idx:end_idx] ** 2)
            c_b[b] = np.sum(c[start_idx:end_idx] * r[start_idx:end_idx] ** 2)

        # ===== COMBINE SPREADING FUNCTION WITH ENERGY AND WEIGHTED PREDICTABILITY =====

        ecb = np.zeros(band_table.shape[0])
        ct = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):
            for bb in range(band_table.shape[0]):
                ecb[b] += spreading_matrix[bb, b] * e_b[bb]
                ct[b] += spreading_matrix[bb, b] * c_b[bb]

        # Normalize the combined energy and weighted predictability

        en = np.zeros(band_table.shape[0])
        cb_b = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):

            if ecb[b] > 0:
                cb_b[b] = ct[b] / ecb[b]
            else:
                cb_b[b] = 0

            sum_spreading = np.sum(spreading_matrix[:, b])

            if sum_spreading > 0:
                en[b] = ecb[b] / sum_spreading
            else:
                en[b] = 0

        # ===== CALCULATE TONALITY INDEX OF EACH BAND=====

        tb_b = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):

            tb_b[b] = -0.299 - 0.43 * np.log(max(cb_b[b], 1e-10))  # Avoid log of zero
            tb_b[b] = np.clip(tb_b[b], 0, 1)  # Clamp to valid range [0, 1]

        # ===== CALCULATE THE SNR OF EACH BAND =====

        NMT = 6 # Noise Masking Tone 6dB for all bands
        TMN = 18 # Tone Masking Noise 18dB for all bands

        SNR_b = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):
            SNR_b[b] = tb_b[b] * TMN + (1 - tb_b[b]) * NMT

        # ===== CONVERT FROM DB TO ENERGY =====

        bc_b = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):
            bc_b[b] = 10 ** (-SNR_b[b] / 10)

        # ===== CALCULATE THE ENERGY THRESHOLD =====

        nb_b = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):
            nb_b[b] = en[b] * bc_b[b]

        # ===== SCALEFACTOR BANDS =====

        eps = np.finfo(float).eps

        qthr_hat = ((eps * N)* (10 ** ((band_table[:, 5]) / 10))) / 2

        npart_b = np.zeros(band_table.shape[0])

        for b in range(band_table.shape[0]):
            npart_b[b] = max(nb_b[b], qthr_hat[b])

        # ===== CALCULATE THE SIGNAL TO MASK RATIO (SMR) =====

        for b in range(band_table.shape[0]):
            SMR[b, k] = e_b[b] / npart_b[b]

    return SMR
