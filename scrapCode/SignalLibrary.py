import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# audioFile = "Peak3.wav"
# audioFile = "StimmenTest_Norm.wav"
# audioFile = "StimmenTest-St.wav"
audioFile = "stimmen_excerpt_patrick_before.wav"

def main():
    a = WavFile(audioFile)
    a.stereo_to_mono()
    a.L_data = a.normalize(a.L_data)

    a.get_events(a.L_data, 0.02, 0.0008, 100)

    #---------------
    dist = a.distMatrix('seconds')
    rms = a.rmsMatrix()
    g = a.gravEq_M(rms, dist, 6)
    r = a.ratioMatrix()
    G = np.round(g * r)
    G = G.astype(int)
    naz = a.negAboveZero(G)
    print(naz)
    print(naz.astype(int))
    np.savetxt("negAboveZero2.csv", naz.astype(int), delimiter=',')
    # print(naz)
    F = a.columnSum(naz)
    # print(F)
    # plt.plot(naz)
    # plt.show()
    # # ---------------
    # for i, item in enumerate(a.audio_events):
    #     item.offset = F[i]
    # canvas = np.zeros(len(a.data))
    # for obj in a.audio_events:
    #     canvas[obj.startIdx + obj.offset : obj.endIdx + obj.offset] += obj.data
    # # print(F)
    # plt.plot(a.data, 'r')
    # plt.plot(canvas, 'g')
    # plt.show()
    # # plt.plot(canvas, 'b')
    # # plt.show()
    # sf.write("Stimmen_after_full_g6.wav", canvas, a.sampleRate)
     
    
        

class Signal:
    def __init__(self, sampleRate=44100):
        self.sampleRate = sampleRate

class WavFile(Signal):
    def __init__(self, fileName):
        Signal.__init__(self)
        self.fileName = fileName
        self.data = self.readWav()
        self.audio_events = None
        self.L_data = None
        self.R_data = None
    
    def readWav(self):
        wav = sf.read(self.fileName)
        self.sampleRate = wav[1]
        return wav[0]

    def normalize(self, data):
        """
        Normalize an array to values between 1 and -1
        """
        if (np.abs(np.max(data)) > np.abs(np.min(data))):
            norm = (1 / np.abs(np.max(data))) * data
            return norm
        else:
            norm = (1 / np.abs(np.min(data))) * data
            return norm
        
    def stereo_to_mono(self):
        """
        Takes a stereo audio signal and splits it into two mono numpy arrays
        """
        if len(self.data[0]) == 2:
            self.L_data = np.array([L for L, R in self.data])
            self.R_data = np.array([R for L, R in self.data])
        else:
            print("Signal is Mono!")

    def rms(self, signal):
        """
        Returns the total RMS value of a give signal
        """
        rms = np.sqrt(np.mean(signal**2))
        return round(rms, 4)

    def rms_analyze(self, data, windowSize=500):
        """
        Returns a 1D Array of RMS values that spans the audio file length

        data: numpy array containing audio data that is to be processed
        windowSize: number of samples being used to calculate any one RMS value
        """
        rms_array = np.array([])
        for i in range(len(data)-windowSize):
            window = self.rms(abs(data[i:i+windowSize]))
            rms_array = np.append(rms_array, window)
        return rms_array

    def find_events(self, data, atkThresh=0.3, dcyThresh=0.005, windowSize=500):
        """
        This function returns tuple pairs of index values.

        data : numpy array containing audio data that is to be processed
        thresh : rms threshold (0.005 default setting for low noise floor)
        windowSize : number of samples being used to determine the threshold
        """
        event_range = []
        onPeak = False
        s = 0
        e = 0
        for i in range(len(data)-windowSize):
            window = self.rms((abs(data[i:i+windowSize])))
            if window < atkThresh and onPeak == False:
                pass
            else:
                if onPeak == False:
                    s = i
                    onPeak = True
                else:
                    pass
                if window >= dcyThresh:
                    pass
                else:
                    e = i
                    event_range.append((s, e))
                    onPeak = False
        return event_range

    def get_events(self, data, atkThresh=0.005, dcyThresh=0.01, windowSize=500):
        event_ranges = self.find_events(data, atkThresh, dcyThresh, windowSize)
        audio_events = np.array([])
        for s, e in event_ranges:
            obj = AudioEvent(data, s, e)
            audio_events = np.append(audio_events, obj)
        self.audio_events = audio_events  

    def write_audio_events(self, fileName, data):
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
        if data.any() == None:
            print("No audio events present\nFirst run self.get_events()")
        else:
            idx = 0
            for item in data:
                idx += 1
                preFix = "{}_{}.wav".format(fileName, idx)
                # preFix = fileName + "_" + str(i + 1) + ".wav"
                sf.write(preFix, item.data, self.sampleRate)
            print("Done")
    
    def order_rms(self):
        h2l = []
        l2h = sorted([obj.eventRms for obj in self.audio_events])
        for i in range(len(l2h)):
            d = len(l2h) - (i+1)
            h2l.append(l2h[d])
        return h2l
    
    def gravEq(self, m1, m2, dist, gconst=1, rPow=2):
        m = m1 * m2
        d = round(dist / self.sampleRate, 4) # convert samples into seconds
        rSqr = d**rPow
        g = gconst * (m / rSqr)
        g *= self.sampleRate #conversion back into samples
        return int(g) # give me an integer of the number of samples m2 will be shifted

    def gravEq_M(self, mM, distM, gconst=1, rPow=2):
        g = np.divide(mM, (distM**rPow), out=np.zeros_like(mM), where=distM!=0) * self.sampleRate
        g = np.multiply(g.astype(int), gconst)
        return g
        

    def distMatrix(self, rType='samples', nDigit=4):
        """
        distMatrix returns a matrix of values derived from the audioEvents list in the WavFile() class. 
        This method will compare all peakIdx values and return a matrix of distances between each event.

        rType: takes a string to indicate return type of the matrix. Default is 'samples' but 'seconds' is also an option.

        nDigit: when returning seconds nDigit determines rounding to the nearest nDigit. Default = 4
        """
        matrix = np.empty([len(self.audio_events), len(self.audio_events)], dtype=int)
        for i in range(len(self.audio_events)):
            for b in range(len(self.audio_events)):
                matrix[i][b] = self.audio_events[i].peakIdx - self.audio_events[b].peakIdx
        if rType.lower() == 'samples':
            return matrix
        elif rType.lower() == 'seconds':
            matrix = matrix / self.sampleRate
            return np.round(matrix, nDigit)

    def rmsMatrix(self):
        """
        rmsMatrix returns a matrix of values derived from the audioEvents list in the WavFile() class. 
        This method will compare all peakIdx values and return a matrix of distances between each event.
        """
        matrix = np.empty([len(self.audio_events), len(self.audio_events)], dtype=float)
        for i in range(len(self.audio_events)):
            for b in range(len(self.audio_events)):
                matrix[i][b] = self.audio_events[i].eventRms * self.audio_events[b].eventRms
        return np.round(matrix, 5)

    def ratioMatrix(self):
        m = np.empty([len(self.audio_events), len(self.audio_events)])
        for i in range(len(self.audio_events)):
            for b in range(len(self.audio_events)):
                r = 1 / (self.audio_events[i].eventRms + self.audio_events[b].eventRms)
                m[i][b] = round(r * self.audio_events[i].eventRms,2)
        return m

    def negAboveZero(self, matrix):
        aboveZero = True
        for i in range(len(matrix)):
            aboveZero = True
            for b in range(len(matrix)):
                if matrix[i][b] == 0:
                    aboveZero = False
                elif aboveZero == True:
                    matrix[b][i] = matrix[b][i] * -1
        return matrix

    def columnSum(self, matrix):
        return matrix.sum(axis=0)





        
                    
class AudioEvent:
    def __init__(self, audioFile, startIdx, endIdx):
        """
        audioFile: numpy array containing audio data
        startIdx: index of audioFile that represents the beginning of the audio event
        endIdx: index of audioFile that represents the end of the audio event
        """
        self.original_data = audioFile
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.offset = 0;
        self.data = self._get_peak_data()
        self.peakIdx = (np.argmax(self.data) + self.startIdx)
        self.eventRms = self.rms()

    def _get_peak_data(self):
        return self.original_data[self.startIdx:self.endIdx]

    def print_peak_params(self):
        print("---------------------")
        print("Start = {}".format(self.startIdx))
        print("Peak  = {}".format(self.peakIdx))
        print("End   = {}".format(self.endIdx))
        print("RMS   = {}".format(self.eventRms))
        print("---------------------")

    def rms(self):
        rms = np.sqrt(np.mean(self.data**2))
        return round(rms, 4)

    def show_peak(self):
        plt.plot(self.data)
        plt.plot([self.peakIdx - self.startIdx],[self.data[self.peakIdx - self.startIdx]], 'ro')
        plt.show()


if __name__ == "__main__":
    main()

#TODO: Implement envelope detection
#TODO: Implement gravity equation. Add param to AudioEvent class for event offset
