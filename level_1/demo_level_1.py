import sys
import numpy as np
import soundfile as sf
import os

# Path to the audio file 
input_file = os.path.join(os.path.dirname(__file__), '..', 'Material', 'LicordeCalandraca.wav')

# Import Encoder and Decoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Encoder_Decoder import aac_coder_1, i_aac_coder_1

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
    print(f" * Encoded {len(aac_seq_1)} frames *")
    
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
    out_wav = os.path.join(os.path.dirname(__file__), '..', 'Output', 'output_level1.wav')
    
    print("=" * 50)
    print("AAC Level 1 - Demo")
    print("=" * 50)
    
    try:
        SNR = demo_aac_1(input_file, out_wav)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print("Please provide a stereo WAV file (48kHz)")
    except Exception as e:
        print(f"Error: {e}")
