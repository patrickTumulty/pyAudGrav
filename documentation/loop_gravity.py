
import pyAudGrav
import matplotlib.pyplot as plt

io = pyAudGrav.load_example1()

analyzer = pyAudGrav.AudioAnalysis(io.data, io.sample_rate)

env = analyzer.get_env_peak(analyzer.data)

new_signal = analyzer.loop_gravity(analyzer.data, env, numLoops=4, plot=False)

io.writeWav("Example1_after.wav", r)

plt.plot([i + analyzer.rStruct.correction for i in range(len(analyzer.data))],
         [item for item in analyzer.data],
         'grey')
plt.plot(new_signal,
         'red')
plt.plot([item.peakIdx + item.offset + analyzer.rStruct.correction for item in analyzer.audio_events],
         [item.panOffset * 0.01 for item in analyzer.audio_events],
         '.blue')
plt.show()

# loop_gravity.py 