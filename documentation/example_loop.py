
import pyAudGrav

audio_file = "example1_voice.wav"

io = pyAudGrav.AudioIO(audio_file)
sig = pyAudGrav.AudioAnalysis(io.data, io.sample_rate)

env = sig.get_env_peak(sig.data)

gConst = 4          # gravitational constant
number_of_loops = 4 # number of times you want to loop over the data
attack = 0.1        # attack threshold
release = 0.001     # release threshold
panRatio = 5
panThresh = 30
plot = False        # plot each iteration

r = sig.loop_gravity(sig.data, env, attack, release, number_of_loops, gConst, panRatio, panThresh, plot)

# io.writeWav("NewFile.wav", r)