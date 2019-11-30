import AudGrav 
import matplotlib.pyplot as plt 

# ==== IMPORT AUDIO ==============================================================

audio_file = "pt_example1_v3_norm.wav"

io = AudGrav.AudioIO(audio_file)

sig = AudGrav.AudioAnalysis(io.data, io.sample_rate) 

sig.data = sig.normalize(sig.data, 1) # peak normalize audio data (Note: This step is optional)

# ==== CREATE ENVELOPE ===========================================================
#       This section creates an envelope of your imported audio data
#       Recommended to use default settings for this section. 
bunch       = 35  # Affects the resolution of the envelope 
lpf         = 100 # Cut off frequency for low pass filter 
filt_order  = 4   # low pass filter slope 

env = sig.get_env_peak(sig.data, bunch, lpf, filt_order, sig.sample_rate) # get_env_rms() is also an option

# ==== SIGNAL ENVELOPE PLOT ======================================================
#       Uncomment lines below to vew this graph

# plt.title("Envelope over Signal")
# plt.xlabel("Time (samples)")
# plt.ylabel("Amplitude")
# plt.plot(sig.data)
# plt.plot(env)
# plt.show()

# ==== CALCULATE THE AUDIO SHIFTING ==============================================
#       This section is where you can edit the parameters that effect
#       your final composition.
atk_thresh  = 0.02      # atk and rel threshold, see SIGNAL ENVELOPE PLOT above
rel_thresh  = 0.001
gConst      = 5         # how much will the various audio events displace eachother in time 
panRatio    = 20        # compression ratio for panning values
panThresh   = 20        # compresstion threshold for pan values (between 0 - 100)
mag_scale   = 'RMS'     # RMS is recommended, but LUFS is an option

sig.calc_shift(sig.data, env, atk_thresh, rel_thresh, gConst, panRatio, panThresh, mag_scale)

# ==== RECONSTRUCT SIGNAL ========================================================
#       Reconstruct your signal

length = len(sig.data) # get length of original audio signal

R = AudGrav.AudioReconstruct(length, sig.audio_events)
r = R.reconstruct_stereo()

#       Note: To output a mono file instead of a stereo file simply change 
#       reconstruct_stereo() to reconstruct_mono()
# ==== WRITE TO A NEW WAVFILE ====================================================

new_file = "NewWavFile.wav" # write your new audio file to the current working directory. 

io.writeWav(new_file, r) # comment out this line if you don't want this script to write a new file

# ==== PLOT NEW SIGNAL ===========================================================
#       This plot allows you to see the timeline of your new signal overlayed with your original audio 
#       To view this graph with no shifting simply add applyShift="N" as an argument in the calc_shift() method. 
#       Blue dots in this graph represent panning values. Negative values are panned left and positive values are 
#       Panned right. 
#       Uncomment lines below to view this graph

plt.title("Reconstructed Signal Over Original Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.plot([i + R.correction for i in range(length)],[item for item in sig.data],'grey')
plt.plot(r, 'r')
plt.plot([obj.peakIdx + obj.offset +  R.correction for obj in sig.audio_events], [obj.panOffset * 0.01 for obj in sig.audio_events], '.b')
plt.show()

#       Note: Amplitude values may seem different than the original signal for the reason 
#       that this plot is showing one channel of the reconstructed audio signal signal 
#       overlayed with the original mono signal. 

