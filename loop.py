import SignalLibrary as sl
import matplotlib.pyplot as plt 

audio_file = "pt_example2.wav"

sig = sl.WavFile(audio_file)
sig.data = sig.normalize(sig.data, 1)
env = sig.get_env_pd(sig.data)

r = sig.loopGravity(sig.data, env, 0.02, 0.001, 10, 2, 2, 20, 40, graph=True)
# plt.stem(sig.panValues)
# plt.show()
# sig.write2wav("pt_example1v2_g4_4loops_r3.wav", r)