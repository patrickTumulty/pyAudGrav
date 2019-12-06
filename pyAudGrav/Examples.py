
from pyAudGrav.AudioIO import AudioIO
from distutils.sysconfig import get_python_lib

audio_file_path = get_python_lib() + "/pyAudGrav/audio_files/"

def load_example1():
    """
    'example1_stimmen.wav' 
    
    This example is an edited excerpt from the electro acoustic 
    composition 'Stimmen' (German: voices) by Dr. Florian Hollerweger.

    This styling of this composition was the original inspiration for 
    creating pyAudGrav.
    """
    return AudioIO(audio_file_path + "example1_stimmen.wav")

def load_example2():
    """
    'example2_tones.wav'

    Simple tones of varying pitch and amplitude, but regular in time. 
    The first four tones are more spaced out than the later four.

    https://freesound.org/ 
    """
    return AudioIO(audio_file_path + "example2_tones.wav")

def load_example3():
    """
    'example3_potsPans.wav'

    Assortment of simple kitchen sounds. 
    """
    return AudioIO(audio_file_path + "example3_potsPans.wav")

def load_example4():
    """
    'example4_pingPong.wav'

    Recording of a dropped ping pong ball. 

    https://freesound.org/
    """
    return AudioIO(audio_file_path + "example4_pingPong.wav")

def load_example5():
    """
    'example5_hey.wav'

    Man saying hey! 
    """
    return AudioIO(audio_file_path + "example5_hey.wav")