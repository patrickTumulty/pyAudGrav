
import soundfile as sf

class AudioIO:
    def __init__(self, fileName):
        self.file_name = fileName
        self.data = self.readWav()

    def readWav(self):
        """
        Return PCM amplitude data in data as type numpy array.

        Automatically updates the sample_rate parameter with sample rate of imported audio.
        """
        wav = sf.read(self.file_name)
        self.sample_rate = wav[1]
        if len(wav[0].shape) == 2:
            channels = 'stereo'
        else:
            channels = 'mono'
        print("Loaded {} | Sample Rate : {} | Channels : {}".format(self.file_name, self.sample_rate, channels))
        return wav[0]

    def write2wav(self, fileName, signal):
        """
        Write numpy array to new wav file. 

        : type fileName : string
        : param fileName : Name of new exported wav file. (Include .wav extension)

        : type signal : Numpy array
        : param signal : Numpy array containing audio data to be written to wav
        """
        if len(signal.shape) == 2:
            channels = 'stereo'
        else:
            channels = 'mono'
        print("Writing {} | Sample Rate : {} | Channels : {}".format(fileName, self.sample_rate, channels))
        sf.write(fileName, signal, self.sample_rate)