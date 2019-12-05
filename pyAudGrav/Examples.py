
from pyAudGrav.AudioIO import AudioIO
from distutils.sysconfig import get_python_lib


def readEx1():
    audio_file_path = get_python_lib +  "pyAudGrav/AudioExamples"
    return AudioIO(audio_file_path + "/Peak3.wav")

