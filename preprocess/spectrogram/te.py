import IPython
import time
import librosa
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile
import scipy.signal
from datetime import timedelta as td
import pandas as pd 
import glob

wav_loc = '/home/gnlenfn/doc/IEMOCAP/Session2/sentences/wav/Ses02F_impro04/Ses02F_impro04_F004.wav'
rate, data = wavfile.read(wav_loc)

lab = pd.read_csv("/home/gnlenfn/remote/lstm/emotion_classes.csv")
count = 0
mel_spec = pd.DataFrame()
for file in glob.glob(wav_loc):
    sr = librosa.get_samplerate(file)
    frame_length = 0.02
    frame_stride = 0.01
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))
    label = np.array(lab.EMOTION[lab.name == file.split("/")[-1].split(".")[0]])
    
    
    stream = librosa.stream(file,
                            block_length=304,
                            frame_length=input_stride,
                            hop_length=input_stride,
                            fill_value=0
                            )
    
    sub = 0
    for y_block in stream:
        sub += 1
        m_block = librosa.stft(y_block, 
                                    n_fft=800,
                                    win_length=input_nfft,
                                    hop_length=input_stride,
                                    window='hamming',
                                    center=False,
                                    #fmax=4000

                                    )
        ab = np.abs(m_block)
        S = librosa.power_to_db(ab**2, ref=np.max)
        norm_S = librosa.util.normalize(S, norm=0)[:200]
        spk = file.split("/")[-1].split(".")[0][0:5]
        rec_type = file.split("/")[-1].split(".")[0][7:12]
        file_name = file.split("/")[-1].split(".")[0] + "-" + str(sub)
        feed_dict = {"file_name":file_name, "speaker":spk, "type":rec_type, "feature":yb, "label":label[0]}
        mel_spec = mel_spec.append(feed_dict, ignore_index=True)