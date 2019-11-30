import numpy as np

class AudioReconstruct:
    def __init__(self, original_length, audio_events):
        """
        Reconstruct audio signal.

        Parameteres:

        original_length : int
            Length, in samples, of the original audio file.

        audio_events : list of type AudioEvent
            List of AudioEvent() objects. 
        """
        self.audio_events = audio_events
        self.bounds_reached = False
        self.correction = 0
        self.original_length = original_length

    def reconstruct_mono(self): 
        """
        Write audio events to a mono numpy array.
        
        Returns a new array.
        """  
        print("Reconstructing MONO signal...")
        new_array = np.zeros(self.original_length)
        for obj in self.audio_events:
            start = obj.startIdx + obj.offset
            new_array = self._add_event_mono(new_array, start, obj.data, 100)
        return new_array

    def reconstruct_stereo(self):
        """
        Write audio events to a stereo numpy array. 

        Returns a new array.
        """
        print("Reconstructing STEREO signal...")
        new_array = np.zeros((self.original_length, 2))
        for obj in self.audio_events:
            L, R = self.pan_trig(obj.panOffset)
            start = obj.startIdx + obj.offset
            new_array = self._add_event_stereo(new_array, start, obj.data, L, R, 100)
        return new_array

    def _add_event_mono(self, array, start, array2, tab=5):
        if start < 0:
            if start < -self.correction:
                prev = self.correction
                self.correction = np.abs(start) + tab
                z = np.zeros(self.correction - prev)
                array = np.insert(array, 0, z)
        elif (len(array2) + start) > self.original_length:
            z = np.zeros((len(array2) + start) - self.original_length + tab)
            array = np.append(array, z)
        array[start + self.correction : start + len(array2) + self.correction] = array2
        return array

    def _add_event_stereo(self, array, start, array2, L, R, tab=5):
        if start < 0:
            if start < -self.correction:
                prev = self.correction
                self.correction = np.abs(start) + tab
                z = np.zeros((self.correction - prev, 2))
                array = np.insert(array, 0, z, 0)
        elif (len(array2) + start) > self.original_length:
            z = np.zeros(((len(array2) + start) - self.original_length + tab, 2))
            array = np.append(array, z, 0)
        for i in range(len(array2)):
            timeline_index = start + self.correction + i
            array[timeline_index][0] += array2[i] * L
            array[timeline_index][1] += array2[i] * R
        return array

    def pan_trig(self, offset):
        """
        Using the Trigonometric Panpot Law, returns a tuple pair of left and right 
        gain.

        Parameter:

        offset : int
            Pan values between -100 (far left) to 100 (far right). 
        """
        degOffset = offset * 0.3
        theta0 = np.pi/2
        theta = np.array([-theta0, self.deg_to_rad(degOffset), theta0])
        gL = np.sin(np.pi*np.abs(theta-theta0)/(4*theta0))
        gR = np.sin(np.pi*(theta+theta0)/(4*theta0))
        return gL[1] , gR[1]

    def deg_to_rad(self, degree):
        """
        convert degrees to radians
        """
        return degree * (np.pi / 180)
    



    

