import sounddevice as sd
import soundfile as sf

print("🎤 Speak for 4 seconds...")
duration = 4
fs = 44100

audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

sf.write("test_voice.wav", audio, fs)
print("✅ Recording saved as test_voice.wav")
