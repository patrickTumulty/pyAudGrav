# pyAudGrav

pyAudGrav is a compositional tool, implemented in python, that allows a user to algorithmically edit and rearrange audio clips, both in time and space, using gravity as a metaphor. 

After the audio file is read, the program will edit out each audio event and treat it as an independent sound object. A sound object (`AudioEvent()`) in this case, is defined as a section of audio that is preceded and followed by the noise floor of the original sound file. The equation of gravity is used as a metaphor to describe the relationship between each audio event. Audio events, based on their mass and distance from other events, will excerpt an attraction. For our purposes, mass is equated as the RMS value of each event and the distance is the time, in seconds squared, in between each events peak index. When actually calculating gravity, we would multiply the equation by the gravitational constant, 9.81 meters per seconds squared, but since audio has no gravitational constant this parameter is exposed to the user to affect the magnitude of shifting. The end result is a new audio file with events that have shifted in time and space (stereo panning) based on mass and distances. 

## Installation

To install use... 

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

io = pyAudGrav.load_example1()                       # pre packaged audio example

analyzer = pyAudGrav.AudioAnalysis(io.data, io.sample_rate)

env = analyzer.get_end_peak(analyzer.data)           # generate envelope 

analyzer.calc_shift(analyzer.data, env, gConst=4)    # calculate shift

rStruct = pyAudGrav.AudioReconstruct(len(analyzer.data), analyzer.audio_events)

new_signal = rStruct.reconstruct_stereo()            # reconstruct signal

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
```

To use your own audio file simply change `io = pyAudGrav.load_example1()` with `io = pyAudGrav.AudioIO(/filePath)`. 

pyAudGrav has a built in function called `loop_gravity()` that allows the user to iterate over the same data set multiple times. This approach yields interesting and different results to that of the example above.

```
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

<<<<<<< HEAD
The code above illustrates the minimum code required to create a new audio file. Examination of the `calc_shift()` and `loop_gravity()` functions will reveal some of the other parameters available to fine tune the pyAudGrav algorithm. 

### `calc_shift()`

```
def calc_shift(self, data, env, atkThresh=0.03, relThresh=0.004, gConst=4, panRatio=5, panThresh=30, magnitudeScale='RMS'):
    """
    Calculates the number of samples that each audio_event element will be shifted. 

    Parameters:

    data : 1D numpy array
        Audio Data

    env : 1D Numpy Array
        Audio Data Envelope.

    atkThresh : float
        Amplitude value for attack threshold (0 < val <= 1).

    relThresh : float
        Amplitude value for release threshold (0 < val <= 1).

    gConst : int or float
        Gravitational constant (Default = 1).

    panRatio : int
        Compression ratio for panning values.
        
    panThresh : int
        Pan compression threshold. (pan values will be normalized after compression)

    magnitudeScale : string
        Define what measure of mass is used to calculate shifting. ('RMS' or 'LUFS')
    """
```

### `loop_gravity()`

```
def loop_gravity(self, data, env, atkThresh=0.03, relThresh=0.004, numLoops=4, gConst=4, panRatio=5, panThresh=30, magnitudeScale='RMS', plot=False):
    """
    Loop over the same audio data multiple times and return the final iteration. 
    
    Parameters:

    data : 1D numpy array
        Audio Data

    env : 1D Numpy Array
        Audio Data Envelope.

    atkThresh : float
        Amplitude value for attack threshold (0 < val <= 1).

    relThresh : float
        Amplitude value for release threshold (0 < val <= 1).

    numLoops : int 
        Number of times to iterate over the data. (Note: numLoops can't be less than 2)

    gConst : int or float
        Gravitational constant (Default = 4).

    panRatio : int
        Compression ratio for panning values.
        
    panThresh : int
        Pan compression threshold. (pan values will be normalized after compression)

    magnitudeScale : string
        Define what measure of mass is used to calculate shifting. ('RMS' or 'LUFS')
        
    plot : boolean
        Plot each iteration. (Default = False)
    """
```



