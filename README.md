# pyAudGrav

pyAudGrav is a compositional tool, implemented in python, that allows a user to algorithmically edit and rearrange audio clips, both in time and space, using gravity. Gravity, in this case, is metaphor used to describe the relationship between sound objects.

![Simple Example (gConst = 2)](/DOCUMENTATION/images/stimmen_shift.png)
*Note: This is a subtle shift example that is meant to illustrate the direction that audio events move in relationship to other events.*

After the audio file is read, the program will edit out each audio event and treat it as an independent sound object. An (`AudioEvent()`) in this case, is defined as a section of audio that is preceded and followed by the noise floor of the original sound file. To approximate the behavior of gravity for these audio events we use Newton's universal law of gravity. The law describes that the force due to gravity is equal to the product between two masses divided by their distance squared and multiplied by a gravitational constant. Audio events, based on their mass and distance from other events, will excerpt an attractive force on other audio events. For our purposes, mass is equated as the RMS value of each event and the distance is the time, in seconds squared, in between each events peak index. When actually calculating gravity, we would multiply the equation by the gravitational constant, 9.81 meters per seconds squared, but since audio has no gravitational constant this parameter is exposed to the user to affect the magnitude of shifting. The end result is a new audio file with events that have shifted in time and space (stereo panning) based on mass and distances. 

![Newtons Law of Gravity](/DOCUMENTATION/images/NewtonsLaw.pdf)

## Installation

To install use... 

`pip install pyAudGrav` 

or

`pip install git+https://github.com/patrickTumulty/pyAudGrav`

## Getting Started 

The package install comes with several example audio files that can be quickly loaded via the following functions. NOTE: `pyAudGrav` currently only supports the importing of MONO files. After the initial analysis the new file can be exported to MONO or STEREO.  

```python
load_example1() # example1_stimmen.wav
load_example2() # example2_tones.wav
load_example3() # example3_potsPans.wav
load_example4() # example4_pingPong.wav
load_example5() # example5_hey.wav
```
These files can also be downloaded directly from the github repository (/DOCUMENTATION/IncludedExamples). To use one of these examples you can use the 
following code. 

```python
import pyAudGrav 
import matplotlib.pyplot as plt

io = pyAudGrav.load_example1()                                 # pre packaged audio example

analyzer = pyAudGrav.AudioAnalysis(io.data, io.sample_rate)    # create an analyzer object 

env = analyzer.get_end_peak(analyzer.data)                     # generate envelope 

analyzer.calc_shift(analyzer.data, env, gConst=4)              # calculate gravity shifting

rStruct = pyAudGrav.AudioReconstruct(len(analyzer.data), 
                                    analyzer.audio_events)     # create reconstruction object

new_signal = rStruct.reconstruct_stereo()                      # reconstruct stereo signal

io.writeWav("Example1_before.wav)                   # original
io.writeWav("Example1_after.wav", new_signal)       # new audio file 

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
```

To use your own audio file simply change `io = pyAudGrav.load_example1()` with `io = pyAudGrav.AudioIO(/filePath)`. 

pyAudGrav has a built in function called `loop_gravity()` that allows the user to iterate over the same data set multiple times. This approach yields interesting and different results to that of the example above. The syntax for achieving this is similar to the previous example with a couple differences. In this example `loop_gravity()` is used in place of `calc_shift()`. In addition `loop_gravity()` will return the final iteration of the looped data. This means that there is no need to create a reconstruction object as is seen in the previous example. Note that `loop_gravity()` will deconstruct and reconstruct the data as a mono data array until the final iteration where it will become stereo. 

```python
import pyAudGrav
import matplotlib.pyplot as plt

io = pyAudGrav.load_example1()

analyzer = pyAudGrav.AudioAnalysis(io.data, io.sample_rate)

env = analyzer.get_env_peak(analyzer.data)

new_signal = analyzer.loop_gravity(analyzer.data, env, numLoops=4, gConst=4, plot=False)

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
```

![Simple Gravity Example](/DOCUMENTATION/images/stimmen_gravity.gif)

The code above illustrates the minimum code required to create a new audio file. Examination of the `calc_shift()` and `loop_gravity()` functions will reveal some of the other parameters available to fine tune the pyAudGrav algorithm. For best results the user is encouraged to experiment with these parameters.

For a more in depth overview and explanation of pyAudGrav and its classes, please refer to the DOCUMENTATION folder above. 

 





