from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np
import time

# audioFile = "stimmen_excerpt_patrick_before.wav"
audioFile = "Peak3.wav"



class Signal:
    def __init__(self, sampleRate=44100):
        self.sampleRate = sampleRate


class SinWave(Signal):
    def __init__(self, f, d):
        """
        Create SinWave 
        f - Frequency in Hertz 
        d - Duraction in Seconds 

        """
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
        """
        Create SinWave 
        f - Frequency in Hertz 
        d - Duraction in Seconds 
        
        """
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
        self.L_data = None
        self.R_data = None
    
    def readWav(self):
        wav = read(self.fileName)
        self.sampleRate = wav[0]
        return wav[1]

    def normalize(self):
        """
        Normalize an array to values between 1 and -1
        """
        if (np.abs(np.max(self.data)) > np.abs(np.min(self.data))):
            norm = (1 / np.abs(np.max(self.data))) * self.data
            self.data = norm
        else:
            norm = (1 / np.abs(np.min(self.data))) * self.data
            self.data = norm
        
    def stereo_to_mono(self):
        """
        Takes a stereo audio signal and splits it into two mono numpy arrays
        """
        if len(self.data[0]) == 2:
            self.L_data = np.array([L for L, R in self.data])
            self.R_data = np.array([R for L, R in self.data])
        else:
            print("Signal is Mono!")


def Windowing(data):
    window = np.array([])
    counter = 0
    total = 0
    for i in range(len(data)):
        if counter < 10:
            window = np.append(window, data[i])
            t = np.sum(np.abs(window)) / 10
            if t < 0.001:
                t = 0
            print(total, ":", t)
            counter += 1
        else:
            window = np.array([])
            counter = 0
            total += 1


def RemoveNoiseFloor(data): 
    """
    Function needs work.

    Currently removes noise floor, but also removes values from 
    the audio signal that we want to keep. 
    """
    for i in range(len(data)):
        if abs(data[i]) < 0.001:
            if data[i] <= 0:
                data[i] = data[i] + abs(data[i])
            else:
                data[i] = data[i] - data[i]
        else:
            pass
    return data


def _peakToEnd(data, idx):
    maxIdx = idx
    signal = np.array([])
    for i in range((len(data) - maxIdx)-5):
        rollingAvg = (sum(abs(data[i + maxIdx : i + maxIdx + 5])) / 5)
        if rollingAvg > 0.001:
            signal = np.append(signal, data[i + maxIdx])
        else:
            print("End :", i + maxIdx)
            break
    return signal

def _peakToBeginning(data, startingIdx):
    maxIdx = startingIdx
    signal = np.array([])
    for i in range((len(data[0:maxIdx])-5)):
        idx = maxIdx - i
        rollingAvg = (sum(abs(data[idx - 5:idx])) / 5)
        if rollingAvg > 0.001:
            signal = np.append(signal, data[idx])
        else:
            print("Beginning: ", idx )
            break
    return signal, idx

def _peak(data, i):
    toEnd = _peakToEnd(data, i)
    p, idx = _peakToBeginning(data, i)
    toBeginning = np.flip(p)
    toBeginning = np.append(toBeginning ,toEnd)
    return toBeginning, idx


def _removePeak(mainData, peakData, startingIdx):
    for i in range(len(peakData)):
        mainData[i + startingIdx] *= 0
    return mainData


def GrabPeaks(audioData, thresh = 0.5):
    """
    Takes a wav data array containing peak events separated by 
    silence and returns a list containing individual arrays for 
    each peak event.
    
    wavData:
    Normalized (values between -1 and 1) wav array.
    Note: Currently doesn't support tuples

    thresh:
    Value above which a peak will be detected. Any value below
    will not be considered a peak.

    """
    events = []
    eventNum = 0
    while (True):
        peakIdx = np.argmax(audioData)
        if (audioData[peakIdx] > thresh):
            p, idx = _peak(audioData, peakIdx)
            events.append(p)
            audioData = _removePeak(audioData, events[eventNum], idx)
            eventNum += 1
        else:
            break
    return events

        
def WriteWaves(events, fileName, sampleRate = 44100):
    """
    WriteWaves takes a list of arrays containing audio events and 
    writes them sequentially to your locally directory as a wav file
    with a given prefix file name. 

    events: 
    A list of arrays containing audio signals 

    fileName:
    A string with the desired fileName. Numbers will be added to 
    delineate different events in your file explorer.  

    """
    for i in range(len(events)):
        preFix = "{}_{}.wav".format(fileName, (i + 1))
        # preFix = fileName + "_" + str(i + 1) + ".wav"
        write(preFix, sampleRate, events[i])
    print("Done")



# a = WavFile(audioFile)
# a.normalize()
# # a.stereo_to_mono()

# e = GrabPeaks(a.data, 0.3)
# WriteWaves(e, 'peak_edited')











        






