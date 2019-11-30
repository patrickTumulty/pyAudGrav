
import AudGrav

io = AudGrav.AudioIO("pt_example2.wav")
sig = AudGrav.AudioAnalysis(io.data, io.sample_rate)
env = sig.get_env_peak(sig.data)
r = sig.loop_gravity(sig.data, env, 0.1, 0.001, 10, 3, plot=True)