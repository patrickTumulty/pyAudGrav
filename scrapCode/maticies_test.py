import numpy as np


idx = np.array([[    0, -18342, -39775],
                [18342,      0, -21433],
                [39775,  21433,      0]])

rms = np.array([[ 0.376,  0.023, 0.0191],
                [ 0.023, 0.1508, 0.0125],
                [0.0191, 0.0125, 0.1034]])

seconds = np.divide(idx, 44100) # convert to seconds
val = np.divide(rms, (seconds**2), out=np.zeros_like(rms), where=seconds!=0) * 44100
val = val.astype(int)

# print("---------------")
# print("IDX   :\n", idx)
# print("---------------")
# print("SEC   :\n", seconds)
# print("---------------")
# print("RMS   :\n", rms)
# print("---------------")
# print("Result:\n", val)

# print((0.023 / ((-18342/44100)**2))*44100)

m2 = 0.1228
m1 = 0.188
r = 0.447
e = -((1 + (m1 / m2))/r**2)
print(e * 44100)
