import os
from pyAudGrav.AudioIO import AudioIO


def readEx1():
    audio_file_path = os.getcwd() + "/AudioExamples"
    return AudioIO(audio_file_path + "/Peak3.wav")

