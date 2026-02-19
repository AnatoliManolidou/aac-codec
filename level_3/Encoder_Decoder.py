import numpy as np
from scipy.io import loadmat
import soundfile as sf
import os
import sys
from scipy.io import savemat

# Import level_1
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'level_1')) 
from SSC import SSC
from Filter_Bank import filter_bank, i_filter_bank

# Import level_2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'level_2')) 
from TNS import tns, i_tns

# Import Huffman Utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Material'))
from huff_utils import encode_huff, decode_huff, load_LUT

# Import Quantization
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Quantization import aac_quantizer, i_aac_quantizer

# Import Psychoacoustic Model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Psychoacoustic_Model import psycho

# Path to the band table file
TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'TableB219.mat')

# Path to the Huffman codebooks file
HUFF_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'huffCodebooks.mat')

def aac_coder_3(filename_in, filename_aac_coded):

    """
    Implements the AAC encoder

    Args:
        filename_in: Input WAV audio file
        filename_aac_coded: .mat file where the AAC coded sequence will be stored

    Returns:
        aac_seq_3: A list of K elements, where K is the number of frames in the input audio. Each element is a dictionary containing the following keys:
            'frame_type': the type of the frame
            'win_type': the window type of the frame
            ['chl']['tns_coeffs']: the TNS coefficients of the left channel, a 4x42 matrix for ESH frames and a 4x69 matrix for the rest types of frames
            ['chr']['tns_coeffs']: the TNS coefficients of the right channel, a 4x42 matrix for ESH frames and a 4x69 matrix for the rest types of frames
            ['chl']['T']: The thresholds of the psychoacoustic model for the left channel, a 42x1 vector for ESH frames and a 69x1 vector for the rest types of frames
            ['chr']['T']: The thresholds of the psychoacoustic model for the right channel, a 42x1 vector for ESH frames and a 69x1 vector for the rest types of frames
            ['chl']['G']: The global gain of the left channel, a scalar for long frames and a 1x8 vector for ESH frames
            ['chr']['G']: The global gain of the right channel, a scalar for long frames and a 1x8 vector for ESH frames
            ['chl']['sfc']: The Huffman encoded sequence of scalefactors for the left channel, a 42x1 vector for ESH frames and a 69x1 vector for the rest types of frames
            ['chr']['sfc']: The Huffman encoded sequence of scalefactors for the right channel, a 42x1 vector for ESH frames and a 69x1 vector for the rest types of frames
            ['chl']['stream']: The Huffman encoded sequence of quantized MDCT coefficients for the left channel, a 1024x1 vector for long frames and an 8x128 vector for ESH frames
            ['chr']['stream']: The Huffman encoded sequence of quantized MDCT coefficients for the right channel, a 1024x1 vector for long frames and an 8x128 vector for ESH frames
            ['chl']['codebook']: The Huffman codebook used for the left channel, a 69x1 vector for long frames and a 42x1 vector for ESH frames
            ['chr']['codebook']: The Huffman codebook used for the right channel, a 69x1 vector for long frames and a 42x1 vector for ESH frames
    
            #ADDITION
            ['chl']['sfc_codebook']: The Huffman codebook used for the scalefactors of the left channel, a scalar value since we are using only one codebook for scalefactors
            ['chr']['sfc_codebook']: The Huffman codebook used for the scalefactors of the right channel, a scalar value since we are using only one codebook for scalefactors
    """
    audio, fs = sf.read(filename_in)
    table = loadmat(TABLE_PATH)
    huff_LUT_list = load_LUT(HUFF_PATH)

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

    # ===== INITIALIZATION =====
    aac_seq_3 = []
    prev_frame_type = 'OLS'  # Start with OLS for the first frame
    win_type = 'KBD'  # Use KBD window for all frames

    # Initialize previous frames 
    frame_T_prev_1 = np.zeros((frame_size, 2))
    frame_T_prev_2 = np.zeros((frame_size, 2))

    # Iterate over each frame
    for i in range(num_frames):

        start_curr = i * hop_size
        end_curr = start_curr + frame_size

        if end_curr + hop_size > audio_padded.shape[0]:
            break

        frame_T = audio_padded[start_curr : end_curr, :]

        start_next = (i + 1) * hop_size
        end_next = start_next + frame_size
        frame_T_next = audio_padded[start_next : end_next, :]

        # ===== SSC =====
        frame_type = SSC(frame_T, frame_T_next, prev_frame_type)

        # ===== MDCT =====
        frame_F = filter_bank(frame_T, frame_type, win_type)

        if frame_type == 'ESH':
            band_table = table['B219b']  # 42 bands for short frames
            num_subframes = 8
            num_coeff = 128
        else:
            band_table = table['B219a']  # 69 bands for long frames
            num_subframes = 1
            num_coeff = 1024

        dict_frame = {
            'frame_type': frame_type,
            'win_type': win_type,
            'chl': {},
            'chr': {}
        }

        # Iterate over channels
        for ch, channel in enumerate(['chl', 'chr']):

            if frame_type == 'ESH':
                frame_F_ch = frame_F[:, :, ch]
            else:
                frame_F_ch = frame_F[:, ch: ch + 1]
                
            frame_F_tns, tns_coeffs = tns(frame_F_ch, frame_type)

            frame_F_tns_1d = frame_F_tns.flatten('F')

            # ===== PSYCHOACOUSTIC MODEL =====
            SMR = psycho(frame_T[:, ch], frame_type, frame_T_prev_1[:, ch], frame_T_prev_2[:, ch])

            # ===== QUANTIZATION =====

            S_k, sfc, G = aac_quantizer(frame_F_tns_1d, frame_type, SMR)

            # ===== THRESHOLD T(b) =====

            T_dict = np.zeros_like(SMR)
            for k in range(num_subframes):
                offset = k * num_coeff
                for b in range(SMR.shape[0]):
                    start_idx = int(band_table[b, 1])
                    end_idx = int(band_table[b, 2]) + 1
                    P_b = np.sum(frame_F_tns_1d[offset + start_idx : offset + end_idx] ** 2)
                    if SMR[b, k] > 0:
                        T_dict[b, k] = P_b / SMR[b, k]
                    else:
                        T_dict[b, k] = 0

            if frame_type != 'ESH':
                T_dict = T_dict[:, 0]  # For long frames, take the first column

            # ===== HUFFMAN ENCODING =====

            # Scale factors encoding (auto-select: uses codebook 11 with ESC for |sfc| > 15)
            sfc_1d = sfc.flatten('F')
            sfc_stream, sfc_codebook = encode_huff(sfc_1d, huff_LUT_list)

            # MDCT coefficients encoding
            codebook_arr = np.zeros((band_table.shape[0], num_subframes), dtype=int)
            stream_bits = []

            for k in range(num_subframes):
                offset = k * num_coeff
                for b in range(SMR.shape[0]):
                    start_idx = int(band_table[b, 1])
                    end_idx = int(band_table[b, 2]) + 1

                    S_band = S_k[offset + start_idx : offset + end_idx]
                
                    stream_band, codebook_band = encode_huff(S_band, huff_LUT_list)
                    stream_bits.append(stream_band)
                    codebook_arr[b, k] = codebook_band

            if frame_type != 'ESH':
                codebook_arr = codebook_arr[:, 0]  # For long frames, take the first column

            # ===== STORE IN DICTIONARY =====
            dict_frame[channel] = {
                'tns_coeffs': tns_coeffs,
                'T': T_dict,
                'G': G,
                'sfc': sfc_stream,
                'sfc_codebook': sfc_codebook,
                'stream': stream_bits,
                'codebook': codebook_arr
            }

            # Update previous frames for the next iteration
            frame_T_prev_2[:, ch] = frame_T_prev_1[:, ch].copy()
            frame_T_prev_1[:, ch] = frame_T[:, ch].copy()

        prev_frame_type = frame_type

        aac_seq_3.append(dict_frame)

    # Save the AAC coded sequence 
    savemat(filename_aac_coded, {'aac_seq_3': aac_seq_3})
    print(f"File saved (aac sequence): {filename_aac_coded}")

    return aac_seq_3

def i_aac_coder_3(aac_seq_3, filename_out):

    """
    Implements the AAC decoder

    Args:
        aac_seq_3: As in the output of aac_coder function
        filename_out: Name of the output WAV audio file

    Returns:
        x: the decoded sequence of samples

    """
    table = loadmat(TABLE_PATH)
    huff_LUT_list = load_LUT(HUFF_PATH)

    frame_size = 2048
    hop_size = 1024
    num_frames = len(aac_seq_3)

    total_samples = num_frames * hop_size + frame_size
    audio_reconstructed = np.zeros((total_samples, 2))

    # Iterate over each frame
    for i in range(num_frames):

        dict_frame = aac_seq_3[i]
        frame_type = dict_frame['frame_type']
        win_type = dict_frame['win_type']

        if frame_type == 'ESH':
            band_table = table['B219b']  # 42 bands for short frames
            num_coeff = 128
            num_subframes = 8
            frame_F = np.zeros((num_coeff, num_subframes, 2))
            
        else:
            band_table = table['B219a']  # 69 bands for long frames
            num_coeff = 1024
            num_subframes = 1
            frame_F = np.zeros((num_coeff, 2))

        # Iterate over channels
        for ch, channel in enumerate(['chl', 'chr']):

            ch_data = dict_frame[channel]

            # ===== HUFFMAN DECODING =====

            # Scale factors decoding
            sfc_stream = ch_data['sfc']
            sfc_cb = ch_data['sfc_codebook']
            sfc_dec_1d = np.array(decode_huff(sfc_stream, huff_LUT_list[sfc_cb]))

            # Trim padding from Huffman decoding because for LONG frames we have 69 sfcs and Huff Encoder has done padding due to encoding in pairs
            expected_len = band_table.shape[0] * num_subframes
            sfc_dec_1d = sfc_dec_1d[:expected_len]

            # Back to 2D shape
            sfc = sfc_dec_1d.reshape((band_table.shape[0], num_subframes), order='F')

            # MDCT coefficients decoding

            stream_list = ch_data['stream']
            codebooks = ch_data['codebook']

            if frame_type != 'ESH':
                codebooks = codebooks[:, np.newaxis]  # For long frames, we add a new axis to make it 2D for consistent indexing
            
            S_k = np.zeros(1024)
            band_idx = 0

            for k in range(num_subframes):
                offset = k * num_coeff

                for b in range(band_table.shape[0]):
                    cb = int(codebooks[b, k]) 
                    start_idx = int(band_table[b, 1])
                    end_idx = int(band_table[b, 2]) + 1

                    if cb > 0:  
                        # Again we trim the padding added by the Huff Encoder for LONG frames 
                        band_len = end_idx - start_idx
                        band_bits = stream_list[band_idx]
                        decoded_band = np.array(decode_huff(band_bits, huff_LUT_list[cb]))
                        S_k[offset + start_idx : offset + end_idx] = decoded_band[:band_len]

                    band_idx += 1

            # ===== INVERSE QUANTIZATION =====
            G = ch_data['G']
            frame_F_tns_1d_decoded = i_aac_quantizer(S_k, sfc, G, frame_type)

            # Back to 2D shape
            if frame_type == 'ESH':
                frame_F_tns_2d = frame_F_tns_1d_decoded.reshape((num_coeff, num_subframes), order='F')
            else:
                frame_F_tns_2d = frame_F_tns_1d_decoded[:, np.newaxis]  # For long frames, we add a new axis to make it 2D for consistent processing

            # ===== INVERSE TNS =====
            tns_coeffs = ch_data['tns_coeffs']
            frame_F_ch = i_tns(frame_F_tns_2d, frame_type, tns_coeffs)

            if frame_type == 'ESH':
                frame_F[:, :, ch] = frame_F_ch
            else:
                frame_F[:, ch] = frame_F_ch[:, 0]

        # Overlap-add
        frame_T = i_filter_bank(frame_F, frame_type, win_type)

        start = i * hop_size
        end = start + frame_size
        audio_reconstructed[start:end, :] += frame_T

    # Padding removal
    valid_start = hop_size
    x = audio_reconstructed[valid_start: , :]

    sf.write(filename_out, x, 48000)
    print(f"File saved (Final audio): {filename_out}")

    return x