import numpy as np
from SignalLibrary.PanPy import PanPy

class Reconstruct:
    def __init__(self):
        pass

    def reconstruct_mono(self, newArray, audioEvents):   
        for obj in audioEvents:
            start = obj.startIdx + obj.offset
            end = obj.endIdx + obj.offset
            newArray[start:end] += obj.data
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
            if mode.upper() == 'T':
                L, R = p.pan_trig(obj.panOffset)
            elif mode.upper() == 'S':
                L, R = p.pan_sqrt(obj.panOffset)
            elif mode.upper() == 'L':
                L, R = p.pan_lin(obj.panOffset)
            for i in range(obj.length):
                timeline_index = (i + obj.startIdx) + obj.offset
                newArray[timeline_index][0] += obj.data[i] * L
                newArray[timeline_index][1] += obj.data[i] * R
        return newArray

    

