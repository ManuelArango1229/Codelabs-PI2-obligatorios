import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os

SRATE = 16000
DUR = 5

print("Escuchando... habla ahora")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Espere, procesando...")
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)
try:
    texto = r.recognize_google(data, language="es-ES")
    print("Dijiste lo siguiente:", texto)
except sr.UnknownValueError:
    print("No se entendió lo que dijiste.")
except sr.RequestError as e:
    print("Ha ocurrido un error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
