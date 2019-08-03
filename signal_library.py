from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np

audioFile = "stimmen_excerpt_patrick_before.wav"


def Count_Peaks(thresh, data):
    count = 0
    peak = True
    for i in range(len(data)):
        if peak:
            if data[i][0] > thresh:
                count += 1
                peak = False
            elif data[i][0] < (thresh * -1):
                count += 1
                peak = False
        elif data[i][0] < thresh and data[i][0] > (thresh * -1):
            peak = True
    return count 


class Signal:
    def __init__(self, sampleRate=44100):
        self.sampleRate = sampleRate

    def plotWave(self):
        plt.plot(self.data)
        plt.xlabel("Time: samples")
        plt.ylabel("Amplitude")
        plt.show()


class SinWave(Signal):
    def __init__(self, f, d):
        Signal.__init__(self)
        self.frequency = f
        self.duration = d
        self.data = self.make_wave()
    
    def make_wave(self):
        time = np.linspace(0, self.duration, (self.duration * self.sampleRate))
        pi2 = 2 * np.pi
        wave = np.sin(pi2 * self.frequency * time)
        return wave


class CosWave(Signal):
    def __init__(self, f, d):
        Signal.__init__(self)
        self.frequency = f
        self.duration = d
        self.data = self.make_wave()
    
    def make_wave(self):
        time = np.linspace(0, self.duration, (self.duration * self.sampleRate))
        pi2 = 2 * np.pi
        wave = np.cos(pi2 * self.frequency * time)
        return wave


class WavFile(Signal):
    def __init__(self, fileName):
        Signal.__init__(self)
        self.fileName = fileName
        self.data = self.readWav()
        
    def readWav(self):
        wav = read(self.fileName)
        return wav[1]

    def normalize(self):
        if (np.abs(np.max(self.data)) > np.abs(np.min(self.data))):
            norm = (1 / np.abs(np.max(self.data))) * self.data
            return norm
        else:
            norm = (1 / np.abs(np.min(self.data))) * self.data
            return norm




        






