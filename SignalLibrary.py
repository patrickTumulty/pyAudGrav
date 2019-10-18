import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import random

audioFile = "Peak3.wav"
# audioFile = "StimmenTest_Norm.wav"
# audioFile = "StimmenTest-St.wav"
# audioFile = "stimmen_excerpt_patrick_before.wav"
# audioFile = "stimmen_excerpt_patrick_before_mono.wav"




def main():
    # sig = WavFile(audioFile)
    # sig.data = sig.normalize(sig.data)
    # env = sig.get_env_pd(sig.data)
    # # sig.get_audio_events(sig.data, env, 0.05, 0.01)
    # sig.calc_shift(sig.data, env, 0.01, 0.001, 10)

    # sig.audio_events[1].print_event_params()
    # val = len(sig.data)
    r = Reconstruct().new_mono(10)
    print(r)
    # n = Reconstruct().reconstruct_stereo(r, sig.audio_events)

    # write('PanTest.wav', sig.sampleRate, r)
    
    # canvas = np.zeros(len(sig.data))
    # sig.calc_shift(sig.data, env, 0.05, 0.001, 4)
    # r = Reconstruct()
    # fR = r.reconstruct_mono(canvas, sig.audio_events)
    # plt.plot(fR)
    # plt.show()
    # sf.write("TestWav.wav", fR, 44100)
    
        

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


class WavFile(Signal):
    def __init__(self, fileName):
        Signal.__init__(self)
        self.fileName = fileName
        self.data = self.readWav()
        self.audio_events = None
        self.L_data = None
        self.R_data = None
    
    def readWav(self):
        """
        Return PCM amplitude data in data as type numpy array.

        Automatically updates the sampleRate parameter with sample rate of imported audio.
        """
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
        This function returns tuple pairs of index values that represent the starting and ending indecies 
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
        audio_events = np.array([])
        for s, e in event_ranges:
            obj = AudioEvent(data, env, s, e)
            audio_events = np.append(audio_events, obj)
        self.audio_events = audio_events  

    def write_audio_events(self, fileName):
        """
        Writes each element of the =audio_events= list into a /.wav/ file. 

        :type fileName: string
        :param fileName: Name of new wav file. (Note: /Writing the .wav extension is not neccessary/)  
        """
        if self.audio_events.any() == None:
            print("No audio events present\nFirst run self.get_audio_events()")
        else:
            idx = 0
            for item in self.audio_events:
                idx += 1
                preFix = "{}_{}.wav".format(fileName, idx)
                # preFix = fileName + "_" + str(i + 1) + ".wav"
                sf.write(preFix, item.data, self.sampleRate)
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
        d = round(dist / self.sampleRate, 4) # convert samples into seconds
        rSqr = d**rPow
        g = gconst * (m / rSqr)
        g *= self.sampleRate #conversion back into samples
        return int(g) # give me an integer of the number of samples m2 will be shifted

    def gravEq_M(self, mM, distM, gconst=1, rPow=2):
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
        g = np.divide(mM, (distM**rPow), out=np.zeros_like(mM), where=distM!=0) * self.sampleRate
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
        Calculates a multiplication table of RMS values derived from the =audio_events= list.

        : type return : numpy matrix astype(float)        
        """
        matrix = np.empty([len(self.audio_events), len(self.audio_events)], dtype=float)
        for i in range(len(self.audio_events)):
            for b in range(len(self.audio_events)):
                matrix[i][b] = self.audio_events[i].eventRms * self.audio_events[b].eventRms
        return np.round(matrix, 5)

    def ratioMatrix(self):
        """
        Calculates a matrix of ratio values comparing each element of the =audio_event= list RMS value. 
        Each pair of audio events will be given a ratio value, the sum of both being 1.

        : type return : numpy matrix astype(float) 
        """
        m = np.empty([len(self.audio_events), len(self.audio_events)])
        for i in range(len(self.audio_events)):
            for b in range(len(self.audio_events)):
                r = 1 / (self.audio_events[i].eventRms + self.audio_events[b].eventRms)
                m[i][b] = round(r * self.audio_events[i].eventRms,2)
        return m

    def panMatrix(self, panScaler=3):
        """
        By comparing the product of two audio events RMS values divided by their distance squared, panMatrix will return an
        array of integer values for the pan offset of each event. In this algorithm larger audio events will cause smaller 
        audio events to be displaced across the stereo image. Smaller audio events will not affect larger audio events. Values
        should then be collapsed via (columnSum) followed by (evaluate_pan())

        """
        mat = np.empty([len(self.audio_events), len(self.audio_events)])
        for i in range(len(self.audio_events)):
            for j in range(len(self.audio_events)):
                if self.audio_events[i].eventRms > self.audio_events[j].eventRms:
                    m = self.audio_events[i].eventRms * self.audio_events[j].eventRms
                    r = round((self.audio_events[i].peakIdx - self.audio_events[j].peakIdx) / 44100, 4)
                    v = m/(r**2)
                else:
                    v = 0
                mat[i][j] = v * (panScaler * 100)
        return mat.astype(int)    

    def evaluate_pan(self, array):
        for i in range(len(array)):
            if array[i] > 100:
                array[i] = 100
            r = random.randrange(0, 2, 1)
            if r == 1:
                array[i] *= -1
        return array

    def negAboveZero(self, matrix):
        """
        Return input matrix with all values above the diagnol zero line as negative integers.

        : type matrix : numpy matrix
        """
        for i in range(len(matrix)):
            for b in range(i):
                matrix[b][i] *= -1
        return matrix

    def columnSum(self, matrix):
        """
        Sum each column of input matrix.

        : type matrix : numpy matrix

        : type return : 1D Numpy array
        """
        return matrix.sum(axis=0)

    def calc_shift(self, data, env, atkThresh, relThresh, gConst=1, rPow=2, panFactor=3, retOption="N", applyShift="Y"):
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
        if self.audio_events == None:
            self.get_audio_events(data, env, atkThresh, relThresh)
        else:
            pass
        self.calc_pan_shift(panFactor)
        d = self.distMatrix('seconds')
        r = self.rmsMatrix()
        g = self.gravEq_M(r, d, gConst, rPow)
        r = self.ratioMatrix()
        G = np.round(g * r)
        G = G.astype(int)
        naz = self.negAboveZero(G)
        shift_amount = self.columnSum(naz)
        if applyShift.upper() == 'Y':
            self._applyShift(shift_amount)
        if retOption.upper() == "N":
            pass
        else:
            return shift_amount 

    def calc_pan_shift(self, panFactor): # TODO: Add to Doc
        pM = self.panMatrix(panFactor)
        cS = self.columnSum(pM)
        eP = self.evaluate_pan(cS)
        self._applyPanOffset(eP)


    def _applyShift(self, shift_array):
        """
        Apply shifted amount to each element in the audio_event list

        : type shiftArray : numpy array astype(int)
        : param shiftArray : Array containing the number of samples that each audio_event element will be shifted 
        """
        for idx, obj in enumerate(self.audio_events):
            obj.offset = shift_array[idx]
    
    def _applyPanOffset(self, pan_array): # TODO: Add to Doc
        """
        Apply pan values to each element in the audio_events list.

        : type pan_array : 

        """
        for idx, obj in enumerate(self.audio_events):
            obj.panOffset = pan_array[idx]
    
    def get_env_pd(self, data, bunch=35, filterCutoff=100, filterOrder=4, sampleRate=44100):
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

        : type sampleRate : int 
        : param sampleRate : Sampling frequency of audio data

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
        b, a = signal.butter(filterOrder, filterCutoff,  fs=sampleRate)
        # filtered = signal.sosfilt(sos, rectified_sig) // this filter method applies delay
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
            
    def write2wav(self, fileName, signal):
        """
        Write numpy array to new wav file. 

        : type fileName : string
        : param fileName : Name of new exported wav file. (Include .wav extension)

        : type signal : Numpy array
        : param signal : Numpy array containing audio data to be written to wav
        """
        write(fileName, self.sampleRate, signal)
                    
class AudioEvent:
    def __init__(self, origData, origEnv, startIdx, endIdx):
        """
        audioFile: numpy array containing audio data
        startIdx: index of audioFile that represents the beginning of the audio event
        endIdx: index of audioFile that represents the end of the audio event
        """
        self.original_data = origData
        self.original_env = origEnv
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.offset = None
        self.panOffset = None
        self.data = self.get_event_data()
        self.peakIdx = np.argmax(self.original_env[self.startIdx:self.endIdx]) + self.startIdx
        self.eventRms = self.rms()

    def get_event_data(self):
        return self.original_data[self.startIdx:self.endIdx]

    def print_event_params(self):
        print("---------------------")
        print("Start    = {}".format(self.startIdx))
        print("Peak     = {}".format(self.peakIdx))
        print("End      = {}".format(self.endIdx))
        print("RMS      = {}".format(self.eventRms))
        print("Offset   = {}".format(self.offset))
        print("Pan      = {}".format(self.panOffset))
        print("---------------------")

    def rms(self):
        rms = np.sqrt(np.mean(self.data**2))
        return round(rms, 4)

    def show_event(self):
        plt.plot(self.data)
        plt.plot(self.original_env[self.startIdx:self.endIdx])
        plt.plot(self.peakIdx-self.startIdx, self.original_env[self.peakIdx], 'ro')
        plt.show()


class Reconstruct:
    def __init__(self):
        pass

    def reconstruct_mono(self, newArray, audioEvents):   
        for obj in audioEvents:
            newArray[obj.startIdx + obj.offset : obj.endIdx + obj.offset] += obj.data
        return newArray

    def new_mono(self, length):
        return np.zeros(length)

    def new_stereo(self, length):
        return np.zeros((length, 2))

    def reconstruct_stereo(self, newArray, audioEvents, mode='T'):
        """
        Write audio data to a multidimensional array with given pan amplitude values. 

        signal: 1D Numpy array
        newArray: Numpy array with shape (length, 2)
        idx: The index value for signal to be inserted
        panVal: A tuple with left and right amplitude values
        """
        p = PanPy()
        for obj in audioEvents:
            for i in range(len(obj.data)):
                if mode.upper() == 'T':
                    L, R = p.pan_trig(obj.panOffset)
                elif mode.upper() == 'S':
                    L, R = p.pan_sqrt(obj.panOffset)
                elif mode.upper() == 'L':
                    L, R = p.pan_lin(obj.panOffset)
                newArray[(i + obj.startIdx) + obj.offset][0] += obj.data[i] * L
                newArray[(i + obj.startIdx) + obj.offset][1] += obj.data[i] * R
        return newArray

    def reconstruct_stereo2(self, signal, newArray, idx, panVal):
        """
        Write audio data to a multidimensional array with given pan amplitude values. 

        signal: 1D Numpy array
        newArray: Numpy array with shape (length, 2)
        idx: The index value for signal to be inserted
        panVal: A tuple with left and right amplitude values
        """
        for i in range(len(signal)):
            newArray[i + idx][0] += signal[i] * panVal[0]
        for i in range(len(signal)):
            newArray[i + idx][1] += signal[i] * panVal[1]
        print("done")
        return newArray


class PanPy:
    def __init__(self):
        pass

    def deg_to_rad(self, degree):
        """
        convert degrees to radians
        """
        return degree * (np.pi / 180)

    def rad_to_deg(self, rad):
        """
        convert radians to degrees
        """
        return round(rad * (180 / np.pi))

    def pan_trig(self, offset):
        """
        Using the Trigonometric Panpot Law, returns a tuple pair of left and right 
        gain.

        : type offset : int
        : param offset : ( -100 <= value <= 100 ) -100 being all the way left and 100 being all the way right
        """
        if offset < 0:
            degOffset = -(offset * 0.9)
        elif offset > 0:
            degOffset = offset * 0.9
        elif offset == 0:
            degOffset = 0
        theta0 = np.pi/2
        theta = np.array([-theta0, self.deg_to_rad(degOffset), theta0])
        gL = np.sin(np.pi*np.abs(theta-theta0)/(4*theta0))
        gR = np.sin(np.pi*(theta+theta0)/(4*theta0))
        return gL[1] , gR[1]

    def pan_sqrt(self, offset):
        """
        Using the Square Root Panpot Law, returns a tuple pair of left and right 
        gain.

        : type offset : int
        : param offset : ( -100 <= value <= 100 ) -100 being all the way left and 100 being all the way right        
        """
        if offset < 0:
            degOffset = -(offset * 0.9)
        elif offset > 0:
            degOffset = offset * 0.9
        elif offset == 0:
            degOffset = 0
        theta0 = np.pi/2
        theta = np.array([-theta0, self.deg_to_rad(degOffset), theta0])
        gL = np.sqrt(np.abs(theta-theta0)/(2*theta0))
        gR = np.sqrt((theta+theta0)/(2*theta0))
        return gL[1] , gR[1]

    def pan_lin(self, offset):
        """
        Using the Linear Panpot Law, returns a tuple pair of left and right 
        gain.

        : type offset : int
        : param offset : ( -100 <= value <= 100 ) -100 being all the way left and 100 being all the way right 
        """
        if offset < 0:
            degOffset = -(offset * 0.9)
        elif offset > 0:
            degOffset = offset * 0.9
        elif offset == 0:
            degOffset = 0
        theta0 = np.pi/2
        theta = np.array([-theta0, self.deg_to_rad(degOffset), theta0])
        gL = np.abs(theta-theta0)/(2*theta0)
        gR = (theta+theta0)/(2*theta0)
        return gL[1] , gR[1]



if __name__ == "__main__":
    main()




