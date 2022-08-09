print("Importing libraries")
import os
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import read, write

inpdir = 'audio-16000'
outdir = 'audio-2s16khz'
sr = 16000
duration = 2 * sr

def centre_of_mass(data):
    total_mass = sum(np.abs(data))
    weighed_dist = sum([i * float(abs(m)) for i, m in enumerate(data)])
    return int((1 / total_mass) * weighed_dist)

def peak(data):
    return np.argmax(np.abs(data))
    
def chunks(data, frames, centre_func=peak):
    min_centre = (frames + 1)//2
    max_centre = len(data)-(frames//2)
    centre = centre_func(data)
    virtual_centre = min(max(min_centre, centre), max_centre)
    return data[virtual_centre-((frames + 1)//2):virtual_centre+(frames//2)]

def main():
    if os.path.exists(outdir) and len(os.listdir(outdir)) > 0:
        print(f"Directory {outdir} already exists and is not empty")
        return 0
    
    os.makedirs(outdir, exist_ok=True)

    print("Finding all audio files")
    filenames = os.listdir(inpdir)

    print("Loading audio files")
    sr, _ = read(os.path.join(inpdir, filenames[0]))
    wavs = np.array([read(os.path.join(inpdir, f))[1] for f in tqdm(filenames)])
    print(f"Shape: {wavs.shape}, dtype: {wavs.dtype}")

    print("Truncating audio files")
    trunc = [chunks(w, duration) for w in tqdm(wavs)]
    trunc = np.array(trunc)
    print(f"Shape: {trunc.shape}, dtype: {trunc.dtype}")

    print("Writing resampled audio data")
    for t, f in tqdm(zip(trunc, filenames)):
        write(os.path.join(outdir, f), sr, t)

##########

if __name__ == "__main__":
    main()
