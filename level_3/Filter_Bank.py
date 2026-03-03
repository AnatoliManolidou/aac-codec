import numpy as np
from scipy.signal.windows import kaiser 

def filter_bank(frame_T, frame_type, win_type):
    """
    Implements the Filterbank stage using MDCT.

    Args:
        frame_T: 2048x2 matrix (current frame, time domain, stereo)
        frame_type: Frame type for the current frame
        win_type: Window type ('KBD' or 'SIN')

    Returns:
        frame_F: MDCT coefficients (shape depends on frame type)
            - For 'OLS', 'LSS', 'LPS': shape (1024, 2)
            - For 'ESH': shape (128, 8, 2)
    """
    
    # ===== LONG FRAMES (OLS, LSS, LPS) =====
    if frame_type in ['OLS', 'LSS', 'LPS']:
        N_long = 2048
        M_long = N_long // 2
        frame_F = np.zeros((M_long, 2))

        # ===== CREATE LONG WINDOW (Wl) =====
        if win_type == 'KBD':
            alpha_long = 6
            w_long = kaiser(M_long + 1, np.pi * alpha_long)
            cum_sum = np.cumsum(w_long)
            divisor = cum_sum[M_long]
            W_long = np.zeros(N_long)
            for n in range(N_long):
                if n < M_long: W_long[n] = np.sqrt(cum_sum[n] / divisor)
                else: W_long[n] = np.sqrt(cum_sum[N_long - 1 - n] / divisor)
        else: 
            W_long = np.sin(np.pi / N_long * (np.arange(N_long) + 0.5))

        # ===== CREATE SHORT WINDOW (Ws) for LSS/LPS =====
        if frame_type in ['LSS', 'LPS']:
            N_short = 256
            M_short = N_short // 2
            if win_type == 'KBD':
                alpha_short = 4
                w_short = kaiser(M_short + 1, np.pi * alpha_short)
                cum_sum = np.cumsum(w_short)
                divisor = cum_sum[M_short]
                W_short = np.zeros(N_short)
                for n in range(N_short):
                    if n < M_short: W_short[n] = np.sqrt(cum_sum[n] / divisor)
                    else: W_short[n] = np.sqrt(cum_sum[N_short - 1 - n] / divisor)
            else:
                W_short = np.sin(np.pi / N_short * (np.arange(N_short) + 0.5))

        # ===== MDCT PREPARATION / COSINE MATRIX CALCULATION (LONG) =====
        n0 = (N_long / 2 + 1) / 2

        n = np.arange(N_long)
        k = np.arange(M_long)[:, np.newaxis]
        cos_matrix = np.cos((2 * np.pi / N_long) * (n + n0) * (k + 0.5))

        # ===== PROCESS EACH CHANNEL =====
        for ch in range(2):
            s = frame_T[:, ch].copy()

            # Apply windowing
            if frame_type == 'OLS':
                s_windowed = s * W_long
            elif frame_type == 'LSS':
                s_windowed = np.zeros(N_long)
                s_windowed[0:1024] = s[0:1024] * W_long[0:1024]
                s_windowed[1024:1472] = s[1024:1472] * 1.0
                s_windowed[1472:1600] = s[1472:1600] * W_short[128:256]
            elif frame_type == 'LPS':
                s_windowed = np.zeros(N_long)
                s_windowed[448:576] = s[448:576] * W_short[0:128]
                s_windowed[576:1024] = s[576:1024] * 1.0
                s_windowed[1024:2048] = s[1024:2048] * W_long[1024:2048]

            # MDCT 
            frame_F[:, ch] = 2.0 * np.dot(cos_matrix, s_windowed)

    # ===== SHORT FRAMES (ESH) =====
    else:
        N_short = 256
        M_short = 128
        frame_F = np.zeros((M_short, 8, 2))

        # ===== CREATE SHORT WINDOW (Ws) =====
        if win_type == 'KBD':
            alpha_short = 4
            w_short = kaiser(M_short + 1, np.pi * alpha_short)
            cum_sum = np.cumsum(w_short)
            divisor = cum_sum[M_short]
            W_short = np.zeros(N_short)
            for n in range(N_short):
                if n < M_short: W_short[n] = np.sqrt(cum_sum[n] / divisor)
                else: W_short[n] = np.sqrt(cum_sum[N_short - 1 - n] / divisor)
        else:
            W_short = np.sin(np.pi / N_short * (np.arange(N_short) + 0.5))

        # ===== MDCT PREPARATION (SHORT) =====
        n0 = (N_short / 2 + 1) / 2

        n = np.arange(N_short)
        k = np.arange(M_short)[:, np.newaxis]
        cos_matrix = np.cos((2 * np.pi / N_short) * (n + n0) * (k + 0.5))

        # ===== PROCESS EACH CHANNEL & SUBFRAME =====
        for ch in range(2):
            for sub in range(8):
                start = 448 + sub * 128
                end = start + N_short
                s_sub = frame_T[start:end, ch] * W_short

                # MDCT with factor 2
                frame_F[:, sub, ch] = 2.0 * np.dot(cos_matrix, s_sub)

    return frame_F

def i_filter_bank(frame_F, frame_type, win_type):
    """
    Implements the Inverse Filterbank stage using IMDCT.

    Args:
        frame_F: MDCT coefficients (shape depends on frame type)
            - For 'OLS', 'LSS', 'LPS': shape (1024, 2)
            - For 'ESH': shape (128, 8, 2)
        frame_type: Frame type for the current frame
        win_type: Window type ('KBD' or 'SIN')

    Returns:
        frame_T: Reconstructed time-domain frame (2048x2 matrix)
    """
    
    frame_T = np.zeros((2048, 2))

    # ===== LONG FRAMES (OLS, LSS, LPS) =====
    if frame_type in ['OLS', 'LSS', 'LPS']:
        N_long = 2048
        M_long = 1024

        # ===== CREATE LONG WINDOW (Wl) =====
        if win_type == 'KBD':
            alpha_long = 6
            w_long = kaiser(M_long + 1, np.pi * alpha_long)
            cum_sum = np.cumsum(w_long)
            divisor = cum_sum[M_long]
            W_long = np.zeros(N_long)
            for n in range(N_long):
                if n < M_long: W_long[n] = np.sqrt(cum_sum[n] / divisor)
                else: W_long[n] = np.sqrt(cum_sum[N_long - 1 - n] / divisor)

            # ===== CREATE SHORT WINDOW (Ws) =====
            N_short = 256
            M_short = 128
            alpha_short = 4
            w_short = kaiser(M_short + 1, np.pi * alpha_short)
            cum_sum = np.cumsum(w_short)
            divisor = cum_sum[M_short]
            W_short = np.zeros(N_short)
            for n in range(N_short):
                if n < M_short: W_short[n] = np.sqrt(cum_sum[n] / divisor)
                else: W_short[n] = np.sqrt(cum_sum[N_short - 1 - n] / divisor)
        else:
            W_long = np.sin(np.pi / N_long * (np.arange(N_long) + 0.5))
            W_short = np.sin(np.pi / 256 * (np.arange(256) + 0.5))

        # ===== IMDCT PREPARATION (LONG) =====
        n0 = (N_long / 2 + 1) / 2

        n = np.arange(N_long)[:, np.newaxis]
        k = np.arange(M_long)
        cos_matrix = np.cos((2 * np.pi / N_long) * (n + n0) * (k + 0.5))

        # ===== PROCESS EACH CHANNEL =====
        for ch in range(2):
            s_reconstructed = (2.0 / N_long) * np.dot(cos_matrix, frame_F[:, ch])
            
            if frame_type == 'OLS':
                s_windowed = s_reconstructed * W_long
            elif frame_type == 'LSS':
                s_windowed = np.zeros(N_long)
                s_windowed[0:1024] = s_reconstructed[0:1024] * W_long[0:1024]
                s_windowed[1024:1472] = s_reconstructed[1024:1472] * 1.0
                s_windowed[1472:1600] = s_reconstructed[1472:1600] * W_short[128:256]
            elif frame_type == 'LPS':
                s_windowed = np.zeros(N_long)
                s_windowed[448:576] = s_reconstructed[448:576] * W_short[0:128]
                s_windowed[576:1024] = s_reconstructed[576:1024] * 1.0
                s_windowed[1024:2048] = s_reconstructed[1024:2048] * W_long[1024:2048]
            
            frame_T[:, ch] = s_windowed

    # ===== SHORT FRAMES (ESH) =====
    else:
        N_short = 256
        M_short = 128

        # ===== CREATE SHORT WINDOW (Ws) =====
        if win_type == 'KBD':
            alpha_short = 4
            w_short = kaiser(M_short + 1, np.pi * alpha_short)
            cum_sum = np.cumsum(w_short)
            divisor = cum_sum[M_short]
            W_short = np.zeros(N_short)
            for n in range(N_short):
                if n < M_short: W_short[n] = np.sqrt(cum_sum[n] / divisor)
                else: W_short[n] = np.sqrt(cum_sum[N_short - 1 - n] / divisor)
        else:
            W_short = np.sin(np.pi / N_short * (np.arange(N_short) + 0.5))

        # ===== IMDCT PREPARATION (SHORT) =====
        n0 = (N_short / 2 + 1) / 2

        n = np.arange(N_short)[:, np.newaxis]
        k = np.arange(M_short)
        cos_matrix = np.cos((2 * np.pi / N_short) * (n + n0) * (k + 0.5))

        # ===== PROCESS EACH CHANNEL & SUBFRAME =====
        for ch in range(2):
            for sub in range(8):
                s_sub_reconstructed = (2.0 / N_short) * np.dot(cos_matrix, frame_F[:, sub, ch])
                s_sub_windowed = s_sub_reconstructed * W_short
                
                start = 448 + sub * 128
                end = start + N_short
                frame_T[start:end, ch] += s_sub_windowed

    return frame_T