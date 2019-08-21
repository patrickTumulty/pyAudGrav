import numpy as np
import signal_library as sl

audio = sl.WavFile("Peak1.wav")
audio.normalize()



def find_all_peaks(data, thresh=0.5):
    vals = []
    for i in range(len(data)-1):
        if data[i+1] < thresh:
            pass
        else:
            if data[i-1] < data[i] and data[i] > data[i + 1]:
                vals.append((i, data[i]))
    return vals

def compare(original, new):
    vals = []
    for i in range(len(original)):
        for b in range(len(new)):
            if original[i][1] == new[b][1]:
                vals.append((original[i][0], new[b][1]))
    return vals

val = find_all_peaks(audio.data)

val2 = find_all_peaks([x[1] for x in val])

val3 = find_all_peaks([x[1] for x in val2])



g = compare(val, val2)
g2 = compare(val, val3)

x_val = [x[0] for x in val]
y_val = [x[1] for x in val]

x_val2 = [x[0] for x in g]
y_val2 = [x[1] for x in g]

x_val3 = [x[0] for x in g2]
y_val3 = [x[1] for x in g2]

sl.plt.plot(audio.data)
sl.plt.plot(x_val, y_val)
sl.plt.plot(x_val2, y_val2)
sl.plt.plot(x_val3, y_val3, ".")
sl.plt.show()
