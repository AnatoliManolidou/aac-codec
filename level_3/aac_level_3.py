import numpy as np
from scipy.io import loadmat
import soundfile as sf
import os
import sys
from scipy.io import savemat

# Import level_1
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'level_1')) 
from aac_level_1 import SSC, filter_bank, i_filter_bank

# Import level_2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'level_2')) 
from aac_level_2 import tns, i_tns

# Import Huffman Utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Material'))
from huff_utils import encode_huff, decode_huff, load_LUT

# Path to the band table file
TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'TableB219.mat')

# Path to the audio file 
input_file = os.path.join(os.path.dirname(__file__), '..', 'Material', 'LicorDeCalandraca.wav')

# Path to the Huffman codebooks file
HUFF_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'huffCodebooks.mat')

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
            c[w] = (np.sqrt(first_term + second_term)) / (r[w] + np.abs(r_pred[w]))

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

    S_k = np.zeros(frame_F.shape[0])
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

                X_hat_band = np.sign(S_band) * (np.abs(S_band) ** (4/3)) * (2**(0.25*a))

                Pe_b = np.sum((X_band - X_hat_band) ** 2)

                if Pe_b < T_b[b]:
                    a += 1

                    if b > 0 and abs(a - alpha[b-1]) > 16:
                        break
                else:
                    break

            alpha[b] = a
            S_k[offset + start_idx : offset + end_idx] = S_band

        # ===== CALCULATE THE GLOBAL GAIN =====
        G[k] = alpha[0]

        # ===== CALCULATE THE SCALE FACTORS =====

        sfc[0, k] = alpha[0]
        for b in range(1, SMR.shape[0]):
            sfc[b, k] = alpha[b] - alpha[b -1]

    if frame_type != 'ESH':
        G = G[0]  # For long frames, G is a scalar

    return S_k, sfc, G


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

            # Scale factors encoding
            sfc_1d = sfc.flatten('F')
            sfc_1d = np.clip(sfc_1d, -16, 16).astype(int)  # Clip to codebook 11 range
            sfc_stream, sfc_codebook = encode_huff(sfc_1d, huff_LUT_list, force_codebook = 11)

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
    print(f"File saved: {filename_aac_coded}")

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
            sfc_dec_1d = np.array(decode_huff(sfc_stream, huff_LUT_list[11]))

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

                        band_bits = stream_list[band_idx]
                        decoded_band = np.array(decode_huff(band_bits, huff_LUT_list[cb]))
                        S_k[offset + start_idx : offset + end_idx] = decoded_band

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
    print(f"File saved: {filename_out}")

    return x

def demo_aac_3(filename_in, filename_out, filename_aac_coded):

    """
    Demonstrates Level 3 AAC coding and decoding

    Args:
        filename_in: Input WAV audio file
        filename_out: Name of the output WAV audio file
        filename_aac_coded: .mat file where the AAC coded sequence will be stored

    Returns:
        SNR: the signal to noise ratio of the reconstructed audio compared to the original audio
        birate: the bitrate of the AAC coded sequence in bits per second
        compression_ratio: bitrate before encoding / bitrate after encoding

    """

    if not os.path.exists(filename_in):
        print(f"Error: File '{filename_in}' not found!")
        return 0

    audio_original, fs = sf.read(filename_in)

    # Encode
    print("Encoding ...")
    aac_seq_3 = aac_coder_3(filename_in, filename_aac_coded)
    print(f"Encoded {len(aac_seq_3)} frames")

    # Decode
    print("Decoding...")
    x = i_aac_coder_3(aac_seq_3, filename_out)
    print(f"Written output file: {filename_out}")

    # ===== SNR CALCULATION =====

    # Checking if the original and reconstructed audio have the same shape
    min_len = min(len(audio_original), len(x))
    audio_ref = audio_original[:min_len, :]
    audio_dec = x[:min_len, :]

    signal_power = np.sum(audio_ref ** 2)
    noise = audio_ref - audio_dec
    noise_power = np.sum(noise ** 2)

    if noise_power < 1e-10:  # Avoid division by zero
        SNR = float('inf')
    else:
        SNR = 10 * np.log10(signal_power / noise_power)

    # ===== BITRATE CALCULATION =====

    # Calculate the total number of bits in the AAC coded sequence
    total_bits = 0
    for frame in aac_seq_3:
        total_bits += 4 # Overhead for frame type and window type

        for channel in ['chl', 'chr']:
            ch_data = frame[channel]
           
            # Bits from Huffman Stream
            for bits in ch_data['stream']:
                total_bits += len(bits)

            # Bits from scalefactors stream
            total_bits += len(ch_data['sfc'])  
        
            # Bits from codebooks
            cb_len = ch_data['codebook'].size
            total_bits += cb_len * 4  # Assuming 4 bits per codebook index

            # Bits from Global Gain
            if np.isscalar(ch_data['G']):
                G_len = 1
            else:
                G_len = len(ch_data['G'])

            total_bits += G_len * 8  # Assuming 8 bits per global gain value

            # Bits from TNS coefficients
            tns_len = ch_data['tns_coeffs'].size
            total_bits += tns_len * 4  # Assuming 4 bits per TNS coefficient

    duration_seconds = len(audio_original) / fs
    bitrate = total_bits / duration_seconds

    # ===== COMPRESSION RATIO CALCULATION =====
    
    original_bitrate = audio_original.shape[1] * fs * 16  # Assuming 16 bits per sample so 48kHz * 16bits * 2 channels
    compression_ratio = original_bitrate / bitrate

    print("=" * 50)
    print(f"Final SNR         = {SNR:.2f} dB")
    print(f"Original Bitrate  = {original_bitrate:.2f} bps")
    print(f"Encoded Bitrate   = {bitrate:.2f} bps")
    print(f"Compression Ratio = {compression_ratio:.2f} : 1")
    print("=" * 50)
    
    return SNR, bitrate, compression_ratio
    
if __name__ == "__main__":
    out_mat = os.path.join(os.path.dirname(__file__), '..', 'Output', 'encoded_seq.mat')
    out_wav = os.path.join(os.path.dirname(__file__), '..', 'Output', 'output_level3.wav')

    print("=" * 50)
    print("AAC Level 3 - Demo")
    print("=" * 50)
    
    try:
        SNR, bitrate, compression = demo_aac_3(input_file, out_wav, out_mat)
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()