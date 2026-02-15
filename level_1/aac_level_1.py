import numpy as np
from scipy.signal import lfilter
import soundfile as sf
import os
from scipy.signal.windows import kaiser 

# Path to the audio file 
input_file = os.path.join(os.path.dirname(__file__), '..', 'Material', 'LicordeCalandraca.wav')

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

        # ===== MDCT PREPARATION (LONG) =====
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

            # MDCT with factor 2
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

def aac_coder_1(filename_in):
    """
    AAC Encoder - Level 1.
    Encodes a stereo audio file into MDCT coefficients.

    Args:
        filename_in: Path to input WAV file

    Returns:
        aac_seq_1: List of K dictionaries, where K is the number of frames.
                   Each dictionary contains:
                   - "frame_type": Frame type ('OLS', 'LSS', 'ESH', 'LPS')
                   - "win_type": Window type ('KBD' or 'SIN')
                   - "chl": Dictionary with "frame_F" (left channel MDCT)
                   - "chr": Dictionary with "frame_F" (right channel MDCT)
    """
    audio, fs = sf.read(filename_in)

    # Verify format
    if fs != 48000: raise ValueError(f"Expected sample rate 48000 Hz, got {fs} Hz")
    if audio.ndim != 2 or audio.shape[1] != 2: raise ValueError("Audio must be stereo")

    frame_size = 2048
    hop_size = 1024  # 50% overlap

    # ===== PADDING =====
    # Zero-pad at the start to handle the first window
    zeros_start = np.zeros((hop_size, 2))
    audio_padded = np.vstack([zeros_start, audio])

    # Zero-pad at the end to complete the last frame
    num_samples = audio_padded.shape[0]
    remainder = (num_samples - frame_size) % hop_size
    if remainder != 0:
        pad_end = hop_size - remainder
        audio_padded = np.vstack([audio_padded, np.zeros((pad_end, 2))])

    # Extra padding for SSC next_frame lookahead
    audio_padded = np.vstack([audio_padded, np.zeros((frame_size, 2))])

    num_frames = (audio_padded.shape[0] - frame_size) // hop_size

    aac_seq_1 = []
    prev_frame_type = 'OLS'
    win_type = 'KBD'

    for i in range(num_frames):
        start_curr = i * hop_size
        end_curr = start_curr + frame_size

        if end_curr + hop_size > audio_padded.shape[0]:
            break

        # Current frame
        frame_T = audio_padded[start_curr:end_curr, :]

        # Next frame for SSC lookahead
        start_next = (i + 1) * hop_size
        end_next = start_next + frame_size
        next_frame_T = audio_padded[start_next:end_next, :]

        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        frame_F = filter_bank(frame_T, frame_type, win_type)

        if frame_type == 'ESH':
            frame_dict = {
                "frame_type": frame_type, "win_type": win_type,
                "chl": {"frame_F": frame_F[:, :, 0]},
                "chr": {"frame_F": frame_F[:, :, 1]}
            }
        else:
            frame_dict = {
                "frame_type": frame_type, "win_type": win_type,
                "chl": {"frame_F": frame_F[:, 0:1]},
                "chr": {"frame_F": frame_F[:, 1:2]}
            }
        
        aac_seq_1.append(frame_dict)
        prev_frame_type = frame_type

    return aac_seq_1


def i_aac_coder_1(aac_seq_1, filename_out):
    """
    AAC Decoder - Level 1.
    Reverses the AACoder1 encoding process.
    
    Args:
        aac_seq_1: List of K dictionaries (encoded frames from aac_coder_1)
        filename_out: Output WAV file path 
    
    Returns:
        x: Decoded audio signal (numpy array, shape: (num_samples, 2))
    """
    frame_size = 2048
    hop_size = 1024
    num_frames = len(aac_seq_1)

    total_samples = num_frames * hop_size + frame_size
    audio_reconstructed = np.zeros((total_samples, 2))

    for i in range(num_frames):
        frame_dict = aac_seq_1[i]
        frame_type = frame_dict["frame_type"]
        win_type = frame_dict["win_type"]

        if frame_type == 'ESH':
            frame_F = np.zeros((128, 8, 2))
            frame_F[:, :, 0] = frame_dict["chl"]["frame_F"]
            frame_F[:, :, 1] = frame_dict["chr"]["frame_F"]
        else:
            frame_F = np.zeros((1024, 2))
            frame_F[:, 0] = frame_dict["chl"]["frame_F"][:, 0]
            frame_F[:, 1] = frame_dict["chr"]["frame_F"][:, 0]

        frame_T = i_filter_bank(frame_F, frame_type, win_type)

        # Overlap-add
        start = i * hop_size
        end = start + frame_size
        audio_reconstructed[start:end, :] += frame_T

    # Discard the initial padding added during encoding
    valid_start = hop_size
    x = audio_reconstructed[valid_start:, :]

    sf.write(filename_out, x, 48000)
    return x


def demo_aac_1(filename_in, filename_out):
    """
    Demonstrates Level 1 AAC encoding/decoding.
    Encodes the input file, decodes it and calculates SNR.
    
    Args:
        filename_in: Input WAV file path
        filename_out: Output WAV file path 
    
    Returns:
        SNR: Signal-to-Noise Ratio in dB
    """

    # Check if file exists
    if not os.path.exists(filename_in):
        print(f"Error: File '{filename_in}' not found!")
        return 0
    
    # Read original audio
    audio_original, fs = sf.read(filename_in)
    
    # Encode using aac_coder_1
    print("Encoding...")
    aac_seq_1 = aac_coder_1(filename_in)
    print(f"Encoded {len(aac_seq_1)} frames")
    
    # Decode using i_aac_coder_1
    print("Decoding...")
    x = i_aac_coder_1(aac_seq_1, filename_out)
    print(f"Written output file: {filename_out}")

    # ===== SNR CALCULATION =====
    min_length = min(len(audio_original), len(x))
    audio_ref = audio_original[:min_length, :]
    audio_dec = x[:min_length, :]

    signal_power = np.sum(audio_ref ** 2)
    noise = audio_ref - audio_dec
    noise_power = np.sum(noise ** 2)
    
    SNR = 10 * np.log10(signal_power / noise_power)
    print("=" * 50)
    print(f"SNR = {SNR:.2f} dB")
    print("Noise power:", noise_power)
    print("=" * 50)
    
    return SNR

# ===== MAIN =====

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), '..', 'Output', 'output_level1.wav')
    
    print("=" * 50)
    print("AAC Level 1 - Demo")
    print("=" * 50)
    
    try:
        SNR = demo_aac_1(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print("Please provide a stereo WAV file (48kHz)")
    except Exception as e:
        print(f"Error: {e}")
