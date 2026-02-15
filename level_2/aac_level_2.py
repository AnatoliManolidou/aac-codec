import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.linalg import solve_toeplitz
import soundfile as sf
import os
import sys

# Import level1
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'level_1')) 
from aac_level_1 import SSC, filter_bank, i_filter_bank

# Path to the band table file
TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'TableB219.mat')

# Path to the audio file 
input_file = os.path.join(os.path.dirname(__file__), '..', 'Material', 'LicorDeCalandraca.wav')

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
            X = frame_F_in[:, sub].copy()
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
        a_iir = np.concatenate(([1.0], -a_quantized))
        X = lfilter([1.0], a_iir, Y)

        if frame_type == 'ESH':
            frame_F_out[:, sub] = X
        else:
            frame_F_out[:, 0] = X

    return frame_F_out


def aac_coder_2(filename_in):
    """
    AAC Encoder - Level 2.
    Encodes a stereo audio file using MDCT + TNS.

    Args:
        filename_in: Path to input WAV file

    Returns:
        aac_seq_2: List of K dictionaries, where K is the number of frames.
                   Each dictionary contains:
                   - "frame_type": Frame type ('OLS', 'LSS', 'ESH', 'LPS')
                   - "win_type": Window type ('KBD' or 'SIN')
                   - "chl": Dict with "frame_F" and "tns_coeffs" (left channel)
                   - "chr": Dict with "frame_F" and "tns_coeffs" (right channel)
    """
    audio, fs = sf.read(filename_in)

     # Verify format
    if fs != 48000: raise ValueError(f"Expected sample rate 48000 Hz, got {fs} Hz")
    if audio.ndim != 2 or audio.shape[1] != 2: raise ValueError("Audio must be stereo")

    frame_size = 2048
    hop_size = 1024  # 50% overlap

    # ===== PADDING =====
    zeros_start = np.zeros((hop_size, 2))
    audio_padded = np.vstack([zeros_start, audio])

    num_samples = audio_padded.shape[0]
    remainder = (num_samples - frame_size) % hop_size
    if remainder != 0:
        pad_end = hop_size - remainder
        audio_padded = np.vstack([audio_padded, np.zeros((pad_end, 2))])

    # Extra padding for SSC next_frame lookahead
    audio_padded = np.vstack([audio_padded, np.zeros((frame_size, 2))])

    num_frames = (audio_padded.shape[0] - frame_size) // hop_size

    aac_seq_2 = []
    prev_frame_type = 'OLS'
    win_type = 'KBD'

    for i in range(num_frames):
        start_curr = i * hop_size
        end_curr = start_curr + frame_size

        if end_curr + hop_size > audio_padded.shape[0]:
            break

        frame_T = audio_padded[start_curr:end_curr, :]

        start_next = (i + 1) * hop_size
        end_next = start_next + frame_size
        next_frame_T = audio_padded[start_next:end_next, :]

        frame_type = SSC(frame_T, next_frame_T, prev_frame_type)
        frame_F = filter_bank(frame_T, frame_type, win_type)

        # ===== TNS PER CHANNEL =====
        if frame_type == 'ESH':
            frame_F_L = frame_F[:, :, 0]  # (128, 8)
            frame_F_R = frame_F[:, :, 1]  # (128, 8)
        else:
            frame_F_L = frame_F[:, 0:1]  # (1024, 1)
            frame_F_R = frame_F[:, 1:2]  # (1024, 1)

        frame_F_L_tns, tns_coeffs_L = tns(frame_F_L, frame_type)
        frame_F_R_tns, tns_coeffs_R = tns(frame_F_R, frame_type)

        frame_dict = {
            "frame_type": frame_type, "win_type": win_type,
            "chl": {"frame_F": frame_F_L_tns, "tns_coeffs": tns_coeffs_L},
            "chr": {"frame_F": frame_F_R_tns, "tns_coeffs": tns_coeffs_R}
        }

        aac_seq_2.append(frame_dict)
        prev_frame_type = frame_type

    return aac_seq_2


def i_aac_coder_2(aac_seq_2, filename_out):
    """
    AAC Decoder - Level 2.
    Reverses the AACoder2 encoding process (inverse TNS + inverse filterbank).

    Args:
        aac_seq_2: List of K dictionaries (encoded frames from AACoder2)
        filename_out: Output WAV file path

    Returns:
        x: Decoded audio signal (numpy array, shape: (num_samples, 2))
    """
    frame_size = 2048
    hop_size = 1024
    num_frames = len(aac_seq_2)

    total_samples = num_frames * hop_size + frame_size
    audio_reconstructed = np.zeros((total_samples, 2))

    for i in range(num_frames):
        frame_dict = aac_seq_2[i]
        frame_type = frame_dict["frame_type"]
        win_type = frame_dict["win_type"]

        # ===== INVERSE TNS PER CHANNEL =====
        frame_F_L = i_tns(frame_dict["chl"]["frame_F"], frame_type, frame_dict["chl"]["tns_coeffs"])
        frame_F_R = i_tns(frame_dict["chr"]["frame_F"], frame_type, frame_dict["chr"]["tns_coeffs"])

        if frame_type == 'ESH':
            frame_F = np.zeros((128, 8, 2))
            frame_F[:, :, 0] = frame_F_L
            frame_F[:, :, 1] = frame_F_R
        else:
            frame_F = np.zeros((1024, 2))
            frame_F[:, 0] = frame_F_L[:, 0]
            frame_F[:, 1] = frame_F_R[:, 0]

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


def demo_aac_2(filename_in, filename_out):
    """
    Demonstrates Level 2 AAC encoding/decoding.
    Encodes the input file with TNS, decodes it, and calculates SNR.

    Args:
        filename_in: Input WAV file path 
        filename_out: Output WAV file path 

    Returns:
        SNR: Signal-to-Noise Ratio in dB
    """
    if not os.path.exists(filename_in):
        print(f"Error: File '{filename_in}' not found!")
        return 0

    audio_original, fs = sf.read(filename_in)

    # Encode
    print("Encoding ...")
    aac_seq_2 = aac_coder_2(filename_in)
    print(f"Encoded {len(aac_seq_2)} frames")

    # Decode
    print("Decoding...")
    x = i_aac_coder_2(aac_seq_2, filename_out)
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
    output_file = os.path.join(os.path.dirname(__file__), '..', 'Output', 'output_level2.wav')

    print("=" * 50)
    print("AAC Level 2 - Demo")
    print("=" * 50)

    try:
        SNR = demo_aac_2(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print("Please provide a stereo WAV file (48kHz)")
    except Exception as e:
        print(f"Error: {e}")