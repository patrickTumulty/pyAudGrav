import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pyAudGrav.AudioEvent import AudioEvent
from pyAudGrav.AudioReconstruct import AudioReconstruct
import pyloudnorm as pyln


class AudioAnalysis:
    def __init__(self, data, sample_rate):
        """
        The WavFile class is intended to handle the the importing, exporting, and analysis a given audio signal.
        Creating an object with given filename (with .wav extension) will automatically import the PCM data. This 
        data is accessible via the =self.data= parameter

        Parameter:

        data : 1D numpy array
            Array containing imported audio data.

        sample_rate : int 
            Sample rate (in Hz) of imported audio data
        """
        self.sample_rate = sample_rate
        self.data = data
        self.audio_events = []

    def normalize(self, data, max_val):
        """
        Normalize an array of data to a defined maximum level. Note: In the case that a 
        signal has positive and negative values, if the absolute value of a negative value
        is greater than the positive maximum, then the method will normalize to that value.

        Parameters:

        data : 1D numpy array
            Values to be normalized. 

        max_val : int
            Maximum value for returned array. 
        """
        if np.abs(np.min(data)) >= np.max(data):
            return data * (max_val/np.abs(np.min(data)))
        elif np.max(data) >= np.abs(np.min(data)):
            return data * (max_val/np.max(data))
        
    def rms(self, signal, roundTo=4):
        """
        Returns the total RMS value of a give signal

        Parameters:

        signal : 1D numpy array
            Audio data. 

        roundTo : int 
            Round return value to the Nth decimal place (Default = 4)
        """
        rms = np.sqrt(np.mean(signal**2))
        return round(rms, roundTo)

    def find_events(self, env, atkThresh, relThresh):
        """
        This function returns tuple pairs of index values that represent the starting and ending indicies 
        of an audio event.

        Parameters:
        
        env : 1D Numpy Array
            Audio Data Envelope.

        atkThresh : float
            Amplitude value for attack threshold (0 < val <= 1).

        relThresh : float
            Amplitude value for release threshold (0 < val <= 1).
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
        Populates the audio_events list. Each element of the list
        is an object of the AudioEvent() class.

        Parameters:

        data : 1D numpy array
            Audio Data

        env : 1D Numpy Array
            Audio Data Envelope.

        atkThresh : float
            Amplitude value for attack threshold (0 < val <= 1).

        relThresh : float
            Amplitude value for release threshold (0 < val <= 1).
        """
        event_ranges = self.find_events(env, atkThresh, relThresh)
        audio_events = np.empty(len(event_ranges), dtype=AudioEvent)
        for idx, item in enumerate(event_ranges):
            obj = AudioEvent(data, env, item[0], item[1], self.sample_rate)
            audio_events[idx] = obj
        self.audio_events = audio_events  

    
    def gravEq_M(self, mM, distM, gconst=1, multSamples='Y'):
        """
        Calculate the gravitational equation using matricies.

        Parameters:

        mM : numpy matrix
            Mass Matrix of RMS values

        distM : numpy matrix
            Distance Matrix

        gconst : int or float
            Gravitational constant (Default = 1).

        return :
            Matrix of samples shifted
        """
        if multSamples.upper() == 'Y':
            g = np.divide(mM, (distM**2), out=np.zeros_like(mM), where=distM!=0) * self.sample_rate
        elif multSamples.upper() == 'N':
            g = np.divide(mM, (distM**2), out=np.zeros_like(mM), where=distM!=0) 
        g = np.multiply(g.astype(int), gconst)
        return g
        
    def distMatrix(self, rType='samples', nDigit=4):
        """
        Calculates a matrix of distances between each audio event. Distances are derived from each audio
        events peak index. 

        Parameters:

        rType : string
            Indicates the return type of the returned matrix (rTypes: 'samples', 'seconds')

        nDigit : int 
            If rType is indicated to be 'seconds' indicate the rounding to the Nth decimal of the returned value.    
        """
        matrix = np.empty([len(self.audio_events), len(self.audio_events)], dtype=int)
        for idx, item in np.ndenumerate(matrix):
            matrix[idx[0]][idx[1]] = self.audio_events[idx[0]].peakIdx - self.audio_events[idx[1]].peakIdx
        if rType.lower() == 'samples':
            return matrix
        elif rType.lower() == 'seconds':
            matrix = matrix / self.sample_rate
            return np.round(matrix, nDigit)

    def rmsMatrix(self):
        """
        Calculates a multiplication table of RMS values.   
        """
        length = len(self.audio_events)
        matrix = np.empty([length, length], dtype=float)
        for idx, item in np.ndenumerate(matrix):
            matrix[idx[0]][idx[1]] = self.audio_events[idx[0]].eventRms * self.audio_events[idx[1]].eventRms
        return np.round(matrix, 5)

    def lufsMatrix(self):
        """
        Calculates a multiplication table of LUFS values. 
        """
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
        """
        matrix = np.empty([len(self.audio_events), len(self.audio_events)])
        for idx, item in np.ndenumerate(matrix):
            r = 1 / (self.audio_events[idx[0]].eventRms + self.audio_events[idx[1]].eventRms)
            matrix[idx[0]][idx[1]] = round(r * self.audio_events[idx[0]].eventRms,2)
        return matrix

    def neg_above_zero(self, matrix, convert):
        """
        Return input matrix with all values above the diagnol zero line as negative integers.
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

    

    def calc_shift(self, data, env, atkThresh, relThresh, gConst=1, panRatio=2, panThresh=50, magnitudeScale='RMS', retOption=False):
        """
        Calculates the number of samples that each audio_event element will be shifted. 

        Parameters:

        data : 1D numpy array
            Audio Data

        env : 1D Numpy Array
            Audio Data Envelope.

        atkThresh : float
            Amplitude value for attack threshold (0 < val <= 1).

        relThresh : float
            Amplitude value for release threshold (0 < val <= 1).

        gConst : int or float
            Gravitational constant (Default = 1).

        panRatio : int
            Compression ratio for panning values.
        
        panThresh : int
            Pan compression threshold. (pan values will be normalized after compression)

        magnitudeScale : string
            Define what measure of mass is used to calculate shifting. ('RMS' or 'LUFS')
        
        retOption : boolean
            Set if the method will return an array of shift amounts. 
        """
        if len(self.audio_events) == 0:
            self.get_audio_events(data, env, atkThresh, relThresh)
        elif len(self.audio_events) != 0:
            self.audio_events = []
            self.get_audio_events(data, env, atkThresh, relThresh)
        d = self.distMatrix('seconds')
        if magnitudeScale == "RMS":
            m = self.rmsMatrix()
            g = self.gravEq_M(m, d, gConst, multSamples='Y')
            r = self.ratioMatrix()
            G = np.round(g * r)
            G = G.astype(int)
            naz = self.neg_above_zero(G, "top")
        elif magnitudeScale == "LUFS":
            m = self.lufsMatrix()
            g = self.gravEq_M(m, d, gConst, multSamples='N')
            r = self.ratioMatrix()
            G = np.round(g * r)
            G = G.astype(int)
            naz = self.neg_above_zero(G, "bottom") 
        shift_amount = naz.sum(axis=0)
        c = self.normalize(shift_amount, 100) # normalize shifting values to 100 and -100
        c = self.compress_pan(c, panRatio, panThresh) 
        self.panValues = self.normalize(c, 100)
        self._apply_pan_offset(self.panValues.astype(int))
        self._apply_shift(shift_amount)
        if retOption == True:
            return shift_amount 

    def compress_pan(self, panValues, compRatio, thresh): 
        """
        Multiply panValues above a set threshold by a ratio. 

        Parameters:

        panValues : 1D Numpy array
            Array of pan values (-100 <= val <= 100).
        
        compRatio : int 
            Compression ratio.
        
        thresh : int 
            Threshold (0 <= val <= 100)
        """
        for i in range(len(panValues)):
            if np.abs(panValues[i]) > thresh:
                panValues[i] *= (1 / compRatio)
        return panValues

    def _apply_shift(self, shift_array):
        """
        Apply shifted amount to each element in the audio_event list
        
        Parameter:

        shiftArray : numpy array astype(int)
            Array containing the number of samples that each audio_event element will be shifted 
        """
        for idx, obj in enumerate(self.audio_events):
            obj.offset = shift_array[idx]
    
    def _apply_pan_offset(self, pan_array): 
        """
        Apply pan values to each element in the audio_events list.

        Parameter:
        
        pan_array : 1D numpy array
            Array of pan values
        """
        for idx, obj in enumerate(self.audio_events):
            obj.panOffset = pan_array[idx]
    
    def get_env_peak(self, data, bunch=35, filterCutoff=100, filterOrder=4, sample_rate=44100):
        """
        Analyze audio data and return an array representing the amplitude envelope of the rectified signal. 
        This method has the advantage of not attenuating the signal. This method will rectify the signal, group
        the samples into bunchs, determine the peak value of each bunch, sum each value in the bunch to that peak value
        and finally apply a LPF to smooth the staircase waveform. Changing the =bunch= and =filterCutoff= arguments will 
        modify the resolution of the returned array.

        Parameters:

        data : 1D numpy array
            Audio data

        bunch : int 
            Window of samples that will be summed to a peak index value (default = 35).

        filterCutoff : int 
            Low pass filter frequency cutoff.

        filterOrder : int 
            Indicated LPF to the Nth order. (Default = 4)
        
        sample_rate : int 
            Sample rate of audio data. 

        return : 1D numpy array
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

        Parameters:

        data : 1D numpy array
            Audio data

        window : int 
            size of window (in samples) used to calculate the RMS envelope. 
        """
        rectified_sig = abs(data)
        rms_array = np.ones(len(data))
        for i in range(len(data) - window):
            val = self.rms(rectified_sig[i:i+window])
            rms_array[i] *= val
        for j in range(len(data) - window, len(data)):
            rms_array[j] = 0
        return rms_array

    def loop_gravity(self, data, env, atkThresh, relThresh, numLoops=2, gConst=1, panRatio=2, panThresh=50, magnitudeScale='RMS', plot=False):
        """
        Loop over the same audio data multiple times and return the final iteration. 
        
        Parameters:

        data : 1D numpy array
            Audio Data

        env : 1D Numpy Array
            Audio Data Envelope.

        atkThresh : float
            Amplitude value for attack threshold (0 < val <= 1).

        relThresh : float
            Amplitude value for release threshold (0 < val <= 1).

        numLoops : int 
            Number of times to iterate over the data. (Note: numLoops can't be less than 2)

        gConst : int or float
            Gravitational constant (Default = 1).

        panRatio : int
            Compression ratio for panning values.
        
        panThresh : int
            Pan compression threshold. (pan values will be normalized after compression)

        magnitudeScale : string
            Define what measure of mass is used to calculate shifting. ('RMS' or 'LUFS')
        
        plot : boolean
            Plot each iteration. (Default = False)
        """
        print("Loop 1")
        self.calc_shift(data, env, atkThresh, relThresh, gConst, panRatio, panThresh, magnitudeScale)
        R = AudioReconstruct(len(self.data), self.audio_events)
        rM = R.reconstruct_mono()
        if plot:
            self.simple_plot(rM)
        if numLoops > 2:
            for i in range(numLoops - 2):
                print("Loop {}".format(i + 2))
                env = self.get_env_peak(rM)
                self.calc_shift(rM, env, atkThresh, relThresh, gConst, panRatio, panThresh, magnitudeScale)
                R = AudioReconstruct(len(rM), self.audio_events)
                rM = R.reconstruct_mono()
                if plot:
                    self.simple_plot(rM)
        print("Loop {}".format(numLoops))
        env = self.get_env_peak(rM)
        self.calc_shift(rM, env, atkThresh, relThresh, gConst, panRatio, panThresh, magnitudeScale)
        R = AudioReconstruct(len(rM), self.audio_events)
        rS = R.reconstruct_stereo()
        if plot:
            self.simple_plot(rS)      
        return rS

    def simple_plot(self, array):
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.plot(array)
        plt.show()

