import wave
import numpy as np
import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def record_full_call(stop_event, filename="full_call_recording.wav"):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening... Speak now.")
    frames = []

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK)
            print("hello")
            frames.append(data)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    filename = "temp_recording.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    return filename