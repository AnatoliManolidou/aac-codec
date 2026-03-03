import numpy as np
import soundfile as sf
import os
import sys

# Path to the audio file 
input_file = os.path.join(os.path.dirname(__file__), '..', 'Material', 'LicorDeCalandraca.wav')

# Import Level 1 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from SSC import SSC
from Filter_Bank import filter_bank, i_filter_bank

# Import TNS 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TNS import tns, i_tns

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
