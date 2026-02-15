import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import soundfile as sf
import os
import sys

# Import level_1
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'level_1')) 
from aac_level_1 import SSC, filter_bank, i_filter_bank

# Import level_2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'level_2')) 
#from aac_level_2 import 

# Path to the band table file
TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Material', 'TableB219.mat')

# Path to the audio file 
input_file = os.path.join(os.path.dirname(__file__), '..', 'Material', 'LicorDeCalandraca.wav')


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
        num_coeffs = 128
        num_subframes = 8
    else:
        band_table = table['B219a']  # 69 bands for long frames
        num_coeffs = 1024
        num_subframes = 1

    # ===== SPREADING FUNCTION CALCULATION =====

    
    

def spreading_function_long(i, j):

    """
    Computes the spreading function value for the given band indices i and j.
    
    Args:
        i: index of the band for which we are calculating the masking threshold
        j: index of the band that is potentially masking band i

    Returns:
        Table value of the spreading function for the given band indices i and j
    """

    table = loadmat(TABLE_PATH)

    band_table = table['B219a']  # 69 bands for long frames
    num_coeffs = 1024
    num_subframes = 1
    bval = band_table[:, 4]  # bval is in the 5th column of the band table
    
    print(band_table[0, :])

    tmpx = 0

    if i >= j:
        #tmpx = 3 * (bval[j] - bval[i])

