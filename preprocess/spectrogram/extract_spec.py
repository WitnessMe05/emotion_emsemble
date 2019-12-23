import glob
import gzip
import os
import pickle

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            print("Directory is already exists")
    except OSError:
        print ('Error: Creating directory. ' +  directory)
lab = pd.read_csv("/home/gnlenfn/remote/lstm/emotion_classes.csv")
count = 0
mel_spec = pd.DataFrame()
for file in glob.glob("/home/gnlenfn/doc/IEMOCAP/*/sentences/wav/*/*.wav"):
    sr = librosa.get_samplerate(file)
    frame_length = 0.02
    frame_stride = 0.01
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))
    label = np.array(lab.EMOTION[lab.name == file.split("/")[-1].split(".")[0]])
    
    
    stream = librosa.stream(file,
                            block_length=309,
                            frame_length=input_stride,
                            hop_length=input_stride,
                            fill_value=0
                            )
    
    sub = 0
    for y_block in stream:
        sub += 1
        m_block = librosa.stft(y_block, 
                                    n_fft=1600,
                                    win_length=input_nfft,
                                    hop_length=input_stride,
                                    window='hamming',
                                    center=False,
                                    #fmax=4000

                                    )
        #print(m_block.shape)
        ab = np.abs(m_block)
        S = librosa.power_to_db(ab**2, ref=np.max)
        norm_S = librosa.util.normalize(S, norm=0)[:400]
        spk = file.split("/")[-1].split(".")[0][0:5]
        rec_type = file.split("/")[-1].split(".")[0][7:12]
        file_name = file.split("/")[-1].split(".")[0] + "-" + str(sub)
        feed_dict = {"file_name":file_name, "speaker":spk, "type":rec_type, "feature":norm_S, "label":label[0]}
        mel_spec = mel_spec.append(feed_dict, ignore_index=True)

    count += 1
    if count % 1000 == 0:
        print("{} / {}".format(count, len(lab)))

print("Extract Done")
mel_spec.to_pickle("/home/gnlenfn/remote/lstm/preprocess/spectrogram/norm_1600_spec.pkl", compression='gzip')
        