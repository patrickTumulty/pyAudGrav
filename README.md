# pyAudGrav

pyAudGrav is a compositional tool, implemented in python, that allows a user to algorithmically edit and rearrange audio clips, both in time and space, using the equation of gravity. 

After the audio file is read, the program, with user-defined attack and release thresholds, will edit out each audio event and treat it as an independent body. An audio event, in this case, is defined as a section of audio that is preceded and followed by the noise floor of the original sound file. The equation of gravity is used to create a relationship between each audio event based on its mass and distance from other events. For our purposes, mass is equated as the RMS value of each event and the distance is the time, in seconds squared, in between each events peak index. When actually calculating gravity, we would multiply the equation by the gravitational constant, 9.81 meters per seconds squared, but since audio has no gravitational constant this parameter is exposed to the user to affect the magnitude of shifting. The end result is a new audio file with events that have shifted in time and space (stereo panning) based on mass and distances. 

# Installation


`pip install git+https://github.com/patrickTumulty/pyAudGrav`

## Example Code

The package comes with several example audio files that can be quickly loaded via the following functions. 

```
load_example1() # example1_stimmen.wav
load_example2() # example2_tones.wav
load_example3() # example3_potsPans.wav
load_example4() # example4_pingPong.wav
load_example5() # example5_hey.wav
```
These files can also be downloaded directly from the github repository. To use one of these examples you can use the 
following code. 

```
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
```

To use your own audio file simply change 'io = pyAudGrav.load_example1' with 'io = pyAudGrav.AudioIO(/filePath)`. 

## Other Usage:

pyAudGrav has a built in function called `loop_gravity()` that allows the user to iterate over the same data set multiple times. This approach yields interesting and different results to that of the example above. 

```
import pyAudGrav

audio_file = "Example.wav"

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

io.writeWav("NewFile.wav", r)
```
