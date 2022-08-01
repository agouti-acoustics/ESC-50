print("Importing libraries")
import os
import numpy as np
from tqdm import tqdm

import librosa
from scipy.io.wavfile import read, write

inpdir = 'audio'
outdir = 'audio-16000'
targsr = 16000

def main():
    if os.path.exists(outdir) and len(os.listdir(outdir)) > 0:
        print(f"Directory {outdir} already exists and is not empty")
        return 0
    
    os.makedirs(outdir, exist_ok=True)

    print("Finding all audio files")
    filenames = os.listdir(inpdir)

    print("Loading audio files")
    origsr, _ = read(os.path.join(inpdir, filenames[0]))
    wavs = [read(os.path.join(inpdir, f))[1] for f in tqdm(filenames)]
    wavs = np.asarray(wavs, dtype=np.float32)
    wavs = wavs / 32768 # 16-bit integer PCM to 32-bit floating-point
    print(f"Shape: {wavs.shape}, dtype: {wavs.dtype}")

    print("Resampling audio files")
    resampled = [librosa.resample(w, orig_sr=origsr, target_sr=targsr) for w in tqdm(wavs)]
    resampled = np.asarray(resampled)
    print(f"Shape: {resampled.shape}, dtype: {resampled.dtype}")

    print("Writing resampled audio data")
    for r, f in tqdm(zip(resampled, filenames)):
        write(os.path.join(outdir, f), targsr, r)

##########

if __name__ == "__main__":
    main()
