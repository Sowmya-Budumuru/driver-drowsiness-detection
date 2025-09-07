# create_beep.py
import numpy as np
from scipy.io.wavfile import write
import os

# Ensure the 'sounds' folder exists
if not os.path.exists("sounds"):
    os.makedirs("sounds")

rate = 44100        # Sampling rate
duration = 1        # seconds
frequency = 1000    # Hz beep sound

t = np.linspace(0, duration, int(rate * duration), endpoint=False)
data = np.sin(2 * np.pi * frequency * t) * 32767
data = data.astype(np.int16)

write("sounds/beep.wav", rate, data)
print("✅ beep.wav generated successfully!")
