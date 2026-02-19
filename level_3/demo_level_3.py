import numpy as np
import soundfile as sf
import os
import sys

# Import Decoder and Encoder 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Encoder_Decoder import aac_coder_3, i_aac_coder_3

# Path to the audio file
input_file = os.path.join(os.path.dirname(__file__), '..', 'Material', 'LicorDeCalandraca.wav')

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
    print(f"* Encoded {len(aac_seq_3)} frames *")

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
        total_bits += 3 # Overhead: frame type (2 bits) + window type (1 bit)

        for channel in ['chl', 'chr']:
            ch_data = frame[channel]
           
            # Bits from Huffman Stream
            for bits in ch_data['stream']:
                total_bits += len(bits)

            # Bits from scalefactors stream
            total_bits += len(ch_data['sfc'])  
        
            # Bits from codebooks

            # MDCT
            mdct_cb_len = ch_data['codebook'].size
            total_bits += mdct_cb_len * 4  # Assuming 4 bits per codebook index

            # scalefactors
            total_bits +=  4  # Assuming 4 bits per scalefactor codebook index

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