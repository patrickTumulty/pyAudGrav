from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np

# audioFile = "stimmen_excerpt_patrick_before.wav"
audioFile = "Peak3_16bit.wav"





class Signal:
    def __init__(self, sampleRate=44100):
        self.sampleRate = sampleRate

    def plotWave(self):
        plt.plot(self.data, ".")
        plt.xlabel("Time: samples")
        plt.ylabel("Amplitude")
        plt.show()

    def Count_Peaks(self, thresh):
        count = 0
        peak = True
        for i in range(len(self.data)):
            if peak:
                if self.data[i] > thresh:
                    count += 1
                    peak = False
                elif self.data[i] < (thresh * -1):
                    count += 1
                    peak = False
            elif self.data[i] < thresh and self.data[i] > (thresh * -1):
                peak = True
        print(count)
        self.peakCount = count 


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
            self.data = norm
        else:
            norm = (1 / np.abs(np.min(self.data))) * self.data
            self.data = norm


def Windowing(data):
    val = np.array([])
    window = np.array([])
    counter = 0
    total = 0
    for i in range(len(data)):
        if counter < 10:
            window = np.append(window, data[i])
            t = np.sum(np.abs(window)) / 10
            if t < 0.001:
                t = 0
            val = np.append(val, t)
            print(total, ":", t)
            counter += 1
        else:
            window = np.array([])
            counter = 0
            total += 1
    plt.plot(val)
    plt.show()




a = WavFile(audioFile)
a.normalize()
Windowing(a.data)

print(np.max(a.data))
print(np.min(a.data))




# Windowing(a.data)

        






