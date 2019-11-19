import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pyln


class AudioEvent:
    def __init__(self, origData, origEnv, startIdx, endIdx, sample_rate):
        """
        The audio event class should be thought of as a container for all applicable data pertaining to one individual
        audio event. An audio event in this case is defined as an isolated, mono, audio signal with a defined beginning and end. 
        Under a normal use case, audio events objects will be automatically created by the =get_audio_events()= method
        in the WavFile() class. 

        : type origData : 1D numpy array
        : param origData : Audio data from imported wav file

        : type origEnv : 1D numpy array
        : param origEnv : envelope of import wav file 

        : type startIdx : int
        : param startIdx : starting index of audio event, within the original import audio data
        
        : type endIdx : int 
        : param endIdx : ending index of audio event, within the original import audio data
        """
        self.sample_rate = sample_rate
        self.original_data = origData
        self.original_env = origEnv
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.offset = 0
        self.panOffset = 0
        self.data = self.get_event_data()
        self.length = len(self.data)
        self.peakIdx = np.argmax(self.original_env[self.startIdx:self.endIdx]) + self.startIdx
        self.eventRms = self.rms()
        self.eventLUFS = self.lufs()

    def get_event_data(self):
        return self.original_data[self.startIdx:self.endIdx]

    def print_event_params(self):
        print("---------------------")
        print("Start    = {}".format(self.startIdx))
        print("Peak     = {}".format(self.peakIdx))
        print("End      = {}".format(self.endIdx))
        print("RMS      = {}".format(self.eventRms))
        print("LUFS     = {}".format(round(self.eventLUFS, 2)))
        print("Offset   = {}".format(self.offset))
        print("Pan      = {}".format(self.panOffset))
        print("---------------------")

    def rms(self):
        rms = np.sqrt(np.mean(self.data**2))
        return round(rms, 4)
    
    def lufs(self):
        l = round(len(self.data) / self.sample_rate, 4)
        if l < 0.4:
            meter = pyln.Meter(self.sample_rate, block_size=l * 0.5)
        else:
            meter = pyln.Meter(self.sample_rate)
        return meter.integrated_loudness(self.data)

    def show_event(self):
        plt.plot(self.data)
        plt.plot(self.original_env[self.startIdx:self.endIdx])
        plt.plot(self.peakIdx-self.startIdx, self.original_env[self.peakIdx], 'ro')
        plt.show()