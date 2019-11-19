import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import random
import scipy.fftpack as sfft
from SignalLibrary.AudioEvent import AudioEvent
from SignalLibrary.Reconstruct import Reconstruct
import pyloudnorm as pyln


class WavFile:
    def __init__(self, fileName):
        """
        The WavFile class is intended to handle the the importing, exporting, and analysis a given audio signal.
        Creating an object with given filename (with .wav extension) will automatically import the PCM data. This 
        data is accessible via the =self.data= parameter

        : type fileName : string
        : param fileName : Name of file (with .wav extension included) to be imported

        """
        self.fileName = fileName
        self.data = self.readWav()
        self.audio_events = []
        self.L_data = None
        self.R_data = None
    
    def readWav(self):
        """
        Return PCM amplitude data in data as type numpy array.

        Automatically updates the sample_rate parameter with sample rate of imported audio.
        """
        wav = sf.read(self.fileName)
        self.sample_rate = wav[1]
        return wav[0]

    def write2wav(self, fileName, signal):
        """
        Write numpy array to new wav file. 

        : type fileName : string
        : param fileName : Name of new exported wav file. (Include .wav extension)

        : type signal : Numpy array
        : param signal : Numpy array containing audio data to be written to wav
        """
        sf.write(fileName, signal, self.sample_rate)

    def normalize(self, data, max_val):
        """
        Normalize an array of data to a defined maximum level. Note: In the case that a 
        signal has positive and negative values, if the absolute value of a negative value
        is greater than the positive maximum, then the method will normalize to that value.

        :type data: numpy array
        :param data: values to be normalized

        :type max_val: int
        :param max_val: maximum value for return array
        """
        if np.abs(np.min(data)) >= np.max(data):
            return data * (max_val/np.abs(np.min(data)))
        elif np.max(data) >= np.abs(np.min(data)):
            return data * (max_val/np.max(data))
        
    def stereo_to_mono(self, method='separate'):
        """
        Takes a stereo audio signal and splits it into two mono numpy arrays
        Two method options 'separate' is going to separate the left and right 
        channels that can be accessed via L_data and R_data or use method 'sum' that
        will sum the left and right channels into a mono signal and assign it to the 
        main data variable.
        """
        if method == 'separate':
            self.L_data = np.array([L for L, R in self.data])
            self.R_data = np.array([R for L, R in self.data])
        elif method == 'sum':
            self.data = np.array([L + R for L, R in self.data])

    def rms(self, signal, roundTo=4):
        """
        Returns the total RMS value of a give signal

        :type signal: 1D Numpy Array
        :param signal: Audio Data

        :type roundTo: int
        :param roundTo: Round return value to the Nth decimal place (Default = 4)
        """
        rms = np.sqrt(np.mean(signal**2))
        return round(rms, roundTo)

    def find_events(self, env, atkThresh, relThresh):
        """
        This function returns tuple pairs of index values that represent the starting and ending indcies 
        of an audio event

        :type env: 1D Numpy Array
        :param env: Audio Data Envelope (See =get_env_pd()= and =get_env_rms()=)

        :type atkThresh: float
        :param atkThresh: amplitude value for attack threshold (0 < val <= 1)

        :type relThresh: float
        :param relThresh: amplitude value for release threshold (0 < val <= 1)
        """
        event_range = []
        onPeak = False
        s = 0
        e = 0
        for i in range(len(env)):
            if env[i] < atkThresh and onPeak == False:
                pass
            else:
                if onPeak == False:
                    s = i
                    onPeak = True
                else:
                    pass
                if env[i] >= relThresh:
                    pass
                else:
                    e = i
                    event_range.append((s, e))
                    onPeak = False
        return event_range

    def get_audio_events(self, data, env, atkThresh, relThresh):
        """
        Populates the =audio_events= list accessible via the =WavFile= object. Each element of the list
        is an object of the =AudioEvent()= class. (See class =AudioEvent=)

        :type data: 1D Numpy Array
        :param data: Audio Data

        :type env: 1D Numpy Array
        :param env: Audio Data Envelope (See =get_env_pd()= and =get_env_rms()=)

        :type atkThresh: float
        :param atkThresh: amplitude value for attack threshold (0 < val <= 1)

        :type relThresh: float
        :param relThresh: amplitude value for release threshold (0 < val <= 1)

        """
        event_ranges = self.find_events(env, atkThresh, relThresh)
        audio_events = np.empty(len(event_ranges), dtype=AudioEvent)
        for idx, item in enumerate(event_ranges):
            obj = AudioEvent(data, env, item[0], item[1], self.sample_rate)
            audio_events[idx] = obj
        self.audio_events = audio_events  

    def write_audio_events(self, fileName):
        """
        Writes each element of the =audio_events= list into a /.wav/ file. 

        :type fileName: string
        :param fileName: Name of new wav file. (Note: /Writing the .wav extension is not neccessary)  
        """
        if self.audio_events.any() == None:
            print("No audio events present\nFirst run self.get_audio_events()")
        else:
            idx = 0
            for item in self.audio_events:
                idx += 1
                preFix = "{}_{}.wav".format(fileName, idx)
                sf.write(preFix, item.data, self.sample_rate)
            print("Done")
    
    def gravEq(self, m1, m2, dist, gconst=1, rPow=2):
        """
        Calculate the gravitational equation to two "bodies of mass" (=AudioEvent=) 

        :type m1: float
        :param m1: RMS value of first "body of mass" (In this case an audio event)

        :type m1: float
        :param m2: RMS value of second "body of mass" (In this case an audio event)

        :type dist: int
        :param dist: Distance (in samples) between m1 and m2

        :type gconst: int
        :param gconst: Gravitational constant (/default=1/)

        :type rPow: int
        :param rPow: Distance to the Nth power (/default=2/)

        :return type: int
        :return param: Number of samples shifted.
        """
        m = m1 * m2
        d = round(dist / self.sample_rate, 4) # convert samples into seconds
        rSqr = d**rPow
        g = gconst * (m / rSqr)
        g *= self.sample_rate #conversion back into samples
        return int(g) # give me an integer of the number of samples m2 will be shifted

    def gravEq_M(self, mM, distM, gconst=1, rPow=2, multSamples='Y'):
        """
        Calculate the gravitational equation using matricies.

        : type mM : numpy matrix
        : param mM : Mass Matrix (see: rmsMatrix())

        : type distM : numpy matrix
        : param distM : Distance Matrix (see: distMatrix())

        : type gconst : int
        : param gconst : Gravitational constant (/default=1/)

        : type rPow : int
        : param rPow : Distance to the Nth power (/default=2/)

        : return type : numpy matrix astype(int)
        : return param : Number of samples shifted.
        """
        if multSamples.upper() == 'Y':
            g = np.divide(mM, (distM**rPow), out=np.zeros_like(mM), where=distM!=0) * self.sample_rate
        elif multSamples.upper() == 'N':
            g = np.divide(mM, (distM**rPow), out=np.zeros_like(mM), where=distM!=0) 
        g = np.multiply(g.astype(int), gconst)
        return g
        
    def distMatrix(self, rType='samples', nDigit=4):
        """
        Calculates a matrix of distances between each audio event. Distances are derived from each audio
        events peak index. Must run =get_audio_events()= first.

        : type rType : string
        : param rType : Indicates the return type of the returned matrix. (rTypes: 'samples', 'seconds')

        : type nDigit : int
        : param nDigit : If rType is indicated to be 'seconds' indicate the rounding to the Nth decimal of the returned value
        """
        matrix = np.empty([len(self.audio_events), len(self.audio_events)], dtype=int)
        for idx, item in np.ndenumerate(matrix):
            matrix[idx[0]][idx[1]] = self.audio_events[idx[0]].peakIdx - self.audio_events[idx[1]].peakIdx
        # for i in range(len(self.audio_events)):
        #     for b in range(len(self.audio_events)):
        #         matrix[i][b] = self.audio_events[i].peakIdx - self.audio_events[b].peakIdx
        if rType.lower() == 'samples':
            return matrix
        elif rType.lower() == 'seconds':
            matrix = matrix / self.sample_rate
            return np.round(matrix, nDigit)

    def rmsMatrix(self):
        """
        Calculates a multiplication table of RMS values derived from the =audio_events= list.

        : type return : numpy matrix astype(float)        
        """
        length = len(self.audio_events)
        matrix = np.empty([length, length], dtype=float)
        for idx, item in np.ndenumerate(matrix):
            matrix[idx[0]][idx[1]] = self.audio_events[idx[0]].eventRms * self.audio_events[idx[1]].eventRms
        return np.round(matrix, 5)

    def lufsMatrix(self):
        length = len(self.audio_events)
        matrix = np.empty([length, length])
        for idx, item in np.ndenumerate(matrix):
            row = self.audio_events[idx[0]].eventLUFS
            col = self.audio_events[idx[1]].eventLUFS
            matrix[idx[0]][idx[1]] = row + col # this is supposed to be multiply, perhaps change
        return matrix

    def ratioMatrix(self):
        """
        Calculates a matrix of ratio values comparing each element of the =audio_event= list RMS value. 
        Each pair of audio events will be given a ratio value, the sum of both being 1.

        : type return : numpy matrix astype(float) 
        """
        matrix = np.empty([len(self.audio_events), len(self.audio_events)])
        for idx, item in np.ndenumerate(matrix):
            r = 1 / (self.audio_events[idx[0]].eventRms + self.audio_events[idx[1]].eventRms)
            matrix[idx[0]][idx[1]] = round(r * self.audio_events[idx[0]].eventRms,2)
        return matrix

    def negAboveZero(self, matrix, convert):
        """
        Return input matrix with all values above the diagnol zero line as negative integers.

        : type matrix : numpy matrix
        """
        if convert.lower() == "top":
            for i in range(len(matrix)):
                for j in range(i):
                    matrix[j][i] *= -1
        elif convert.lower() == "bottom":
            for i in range(len(matrix)):
                for j in range(i):
                    matrix[i][j] *= -1
        return matrix

    def columnSum(self, matrix):
        """
        Sum each column of input matrix.

        : type matrix : numpy matrix
        : type return : 1D Numpy array
        """
        return matrix.sum(axis=0)

    def calc_shift(self, data, env, atkThresh, relThresh, gConst=1, rPow=2, panRatio=2, panThresh=50, magnitudeScale='RMS',retOption="N", applyShift="Y"):
        """
        Calculates the number of samples that each =audio_event= element will be shifted. 

        : type data : 1D Numpy Array
        : param data : audio data

        : type atkThresh : float 
        : param atkThresh : amplitude value for attack threshold (0 < val <= 1) 

        : type relThresh : float
        : param relThresh : amplitude value for release threshold (0 < val <= 1)

        : type gconst : int
        : param gconst : Gravitational constant (/default=1/)

        : type rPow : int
        : param rPow : Distance to the Nth power (/default=2/)
        
        : type panFactor : int
        : param panFactor : Control how drastically audio events are scattered over the stereo image (Values Range: 1 - 10)

        : type retOption : string 
        : param retOption : Changes whether or not the function will return an array of integers for the shift amount for each audio event (default = 'N') 

        : type applyShift : string
        : param applyShift : Changes whether or not the function will automatically apply the shift amount to the corresponding audio event (default = 'Y')
        """
        if len(self.audio_events) == 0:
            self.get_audio_events(data, env, atkThresh, relThresh)
        elif len(self.audio_events) != 0:
            self.audio_events = []
            self.get_audio_events(data, env, atkThresh, relThresh)
        d = self.distMatrix('seconds')
        if magnitudeScale == "RMS":
            m = self.rmsMatrix()
            g = self.gravEq_M(m, d, gConst, rPow, multSamples='Y')
            r = self.ratioMatrix()
            G = np.round(g * r)
            G = G.astype(int)
            naz = self.negAboveZero(G, "top")
        elif magnitudeScale == "LUFS":
            m = self.lufsMatrix()
            g = self.gravEq_M(m, d, gConst, rPow, multSamples='N')
            r = self.ratioMatrix()
            G = np.round(g * r)
            G = G.astype(int)
            naz = self.negAboveZero(G, "bottom")
        # print(naz)
        shift_amount = self.columnSum(naz)
        # print(shift_amount)
        c = self.normalize(shift_amount, 100) # normalize shifting values to 100 and -100
        c = self.compress_pan(c, panRatio, panThresh) 
        self.panValues = self.normalize(c, 100)
        self._applyPanOffset(self.panValues.astype(int))
        if applyShift.upper() == 'Y':
            self._applyShift(shift_amount)
        if retOption.upper() == "N":
            pass
        else:
            return shift_amount 

    def compress_pan(self, panValues, compRatio, thresh): 
        for i in range(len(panValues)):
            if np.abs(panValues[i]) > thresh:
                panValues[i] *= (1 / compRatio)
        return panValues

    def _applyShift(self, shift_array):
        """
        Apply shifted amount to each element in the audio_event list

        : type shiftArray : numpy array astype(int)
        : param shiftArray : Array containing the number of samples that each audio_event element will be shifted 
        """
        for idx, obj in enumerate(self.audio_events):
            obj.offset = shift_array[idx]
    
    def _applyPanOffset(self, pan_array): 
        """
        Apply pan values to each element in the audio_events list.

        : type pan_array : 1D numpy array
        : param pan_array : Array of pan values (int between -100 and 100)  
        """
        for idx, obj in enumerate(self.audio_events):
            obj.panOffset = pan_array[idx]
    
    def get_env_pd(self, data, bunch=35, filterCutoff=100, filterOrder=4, sample_rate=44100):
        """
        Analyze audio data and return an array representing the amplitude envelope of the rectified signal. 
        This method has the advantage of not attenuating the signal. This method will rectify the signal, group
        the samples into bunchs, determine the peak value of each bunch, sum each value in the bunch to that peak value
        and finally apply a LPF to smooth the staircase waveform. Changing the =bunch= and =filterCutoff= arguments will 
        modify the resolution of the returned array.

        : type data : 1D numpy array
        : param data : Audio data

        : type bunch : int 
        : param bunch : Window of samples that will be summed to a peak index value (default = 35)

        : type filterCutoff : int
        : param filterCutoff : Low pass filter frequency cutoff

        : type filterOrder : int 
        : param filterOrder : Indicate LPF to the Nth order. (default=4)

        : type sample_rate : int 
        : param sample_rate : Sampling frequency of audio data

        : type return : 1D Numpy array
        """
        rectified_sig = abs(data)
        rem = len(rectified_sig) % bunch
        for i in range(int(len(rectified_sig)/bunch)):
            i *= bunch
            if i + bunch > len(rectified_sig):
                rectified_sig[i:i+rem] = max(rectified_sig[i:i+rem]) * np.ones(rem)
            else:
                rectified_sig[i:i+bunch] = max(rectified_sig[i:i+bunch]) * np.ones(bunch)
        b, a = signal.butter(filterOrder, filterCutoff,  fs=sample_rate)
        filtered = signal.filtfilt(b, a, rectified_sig, method='gust')
        return filtered
    
    def get_env_rms(self, data, window):
        """
        Analyze audio data and return an array representing the amplitude of the signal. This method uses a moving window of samples that 
        are calculated via RMS. 

        : type data : 1D Numpy Array
        : param data : audio data

        : type window : int 
        : param window : size of window (in samples) used to calculate the RMS envelope
        """
        rectified_sig = abs(data)
        rms_array = np.ones(len(data))
        for i in range(len(data) - window):
            val = self.rms(rectified_sig[i:i+window])
            rms_array[i] *= val
        for j in range(len(data) - window, len(data)):
            rms_array[j] = 0
        return rms_array
  
    def loopGravity(self, data, env, atkThresh, relThresh, numLoops, gConst=1, rPow=2, panRatio=2, panThresh=50, magnitudeScale='RMS', mos='S'):
        self.calc_shift(data, env, atkThresh, relThresh, gConst, rPow, panRatio, panThresh, magnitudeScale, retOption="N", applyShift="Y")
        rM = Reconstruct().new_mono(len(self.data))
        rM = Reconstruct().reconstruct_mono(rM, self.audio_events)
        # plt.plot(data, 'r')
        # plt.plot(rM, 'g')
        # plt.show()
        for i in range(numLoops):
            env = self.get_env_pd(rM)
            self.calc_shift(rM, env, atkThresh, relThresh, gConst, rPow, panRatio, panThresh, magnitudeScale, retOption="N", applyShift="Y")
            rM = Reconstruct().new_mono(len(rM))
            rM = Reconstruct().reconstruct_mono(rM, self.audio_events)
            # plt.plot(data, 'r')
            # plt.plot(rM, 'g')
            # plt.show()
        env = self.get_env_pd(rM)
        self.calc_shift(rM, env, atkThresh, relThresh, gConst, rPow, panRatio, panThresh, magnitudeScale, retOption="N", applyShift="Y")
        rS = Reconstruct().new_stereo(len(rM))
        rS = Reconstruct().reconstruct_stereo(rS, self.audio_events)
        # plt.plot(data, 'r')
        # plt.plot(rS, 'g')
        return rS
        
