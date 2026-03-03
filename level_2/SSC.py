import numpy as np
from scipy.signal import lfilter

def SSC(frame_T, next_frame_T, prev_frame_type):
    """
    Implements the Sequence Segmentation Control stage.
    Determines the frame type ('OLS', 'LSS', 'ESH', 'LPS') for the current frame based on the energy of the next frame.

    Args:
        frame_T: 2048x2 matrix (current frame, time domain, stereo)
        next_frame_T: 2048x2 matrix (next frame, time domain, stereo)
        prev_frame_type: Frame type of the previous frame ('OLS', 'LSS', 'ESH', 'LPS')

    Returns:
        frame_type: Frame type for the current frame (string)
            - 'OLS' for ONLY_LONG_SEQUENCE
            - 'LSS' for LONG_START_SEQUENCE
            - 'ESH' for EIGHT_SHORT_SEQUENCE
            - 'LPS' for LONG_STOP_SEQUENCE
    """
    # Filter coefficients for high-pass filter
    b = [0.7548, -0.7548]
    a = [1, -0.5095]

    has_attack = [False, False]
    channel_decisions = []

    for ch in range(2):
        s_next = next_frame_T[:, ch]

        # Apply high-pass filter to the next frame
        s_filt = lfilter(b, a, s_next)

        # ===== ATTACK DETECTION =====
        for i in range(8):
            start = 448 + i * 128
            end = start + 128

            # Energy of sub-region l: s_l^2
            segment = s_filt[start:end]
            s_l2 = np.sum(segment ** 2)

            # Attack value: d_sl^2
            if i == 0:
                d_sl2 = 0.0
            else:
                sum_prev = 0.0
                for j in range(i):
                    prev_start = 448 + j * 128
                    prev_end = prev_start + 128
                    prev_segment = s_filt[prev_start:prev_end]
                    sum_prev += np.sum(prev_segment ** 2)

                avg_prev = sum_prev / i

                if avg_prev < 1e-10:
                    d_sl2 = 0.0
                else:
                    d_sl2 = s_l2 / avg_prev

            # Attack condition: s_l^2 > 10^-3 AND d_sl^2 > 10
            if (s_l2 > 1e-3) and (d_sl2 > 10):
                has_attack[ch] = True
                break

        # ===== PER-CHANNEL DECISION LOGIC =====
        if prev_frame_type == "LSS":
            channel_decisions.append("ESH")
        elif prev_frame_type == "LPS":
            channel_decisions.append("OLS")
        elif prev_frame_type == "OLS":
            if has_attack[ch]:
                channel_decisions.append("LSS")
            else:
                channel_decisions.append("OLS")
        elif prev_frame_type == "ESH":
            if has_attack[ch]:
                channel_decisions.append("ESH")
            else:
                channel_decisions.append("LPS")
        else:
            if has_attack[ch]:
                channel_decisions.append("LSS")
            else:
                channel_decisions.append("OLS")

    # ===== FINAL FRAME TYPE DECISION =====
    decision_table = {
        ('OLS', 'OLS'): 'OLS',
        ('OLS', 'LSS'): 'LSS',
        ('OLS', 'ESH'): 'ESH',
        ('OLS', 'LPS'): 'LPS',
        ('LSS', 'OLS'): 'LSS',
        ('LSS', 'LSS'): 'LSS',
        ('LSS', 'ESH'): 'ESH',
        ('LSS', 'LPS'): 'ESH',
        ('ESH', 'OLS'): 'ESH',
        ('ESH', 'LSS'): 'ESH',
        ('ESH', 'ESH'): 'ESH',
        ('ESH', 'LPS'): 'ESH',
        ('LPS', 'OLS'): 'LPS',
        ('LPS', 'LSS'): 'ESH',
        ('LPS', 'ESH'): 'ESH',
        ('LPS', 'LPS'): 'LPS',
    }

    frame_type = decision_table[(channel_decisions[0], channel_decisions[1])]

    return frame_type
