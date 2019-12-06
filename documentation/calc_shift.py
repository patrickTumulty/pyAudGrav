import pyAudGrav 
import matplotlib.pyplot as plt

io = pyAudGrav.load_example1() # pre packaged audio example

analyzer = pyAudGrav.AudioAnalysis(io.data, io.sample_rate)

env = analyzer.get_end_peak(analyzer.data) # generate envelope 

analyzer.calc_shift(analyzer.data, env) # calculate shift

rStruct = pyAudGrav.AudioReconstruct(len(analyzer.data), analyzer.audio_events)

new_signal = rStruct.reconstruct_stereo() # reconstruct signal

io.writeWav("Example1_before.wav)
io.writeWav("Example1_after.wav", new_signal)

# == Plot New Signal == 

plt.title("Reconstructed Signal Over Original Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.plot([i + rStruct.correction for i in range(len(analyzer.data))],
         [item for item in analyzer.data],
         'grey')
plt.plot(new_signal, 'r')
plt.plot([obj.peakIdx + obj.offset + rStruct.correction for obj in analyzer.audio_events], 
         [obj.panOffset * 0.01 for obj in analyzer.audio_events], 
         '.b')
plt.show()

# calc_shift.py 