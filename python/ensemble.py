import numpy as np 
import tensorflow as tf 
import glob, gzip, pickle
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

def stp(char):
    return char.strip("'")

def build_lstm():
    inputs = tf.keras.Input(shape=(1,200,300))
    conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu)(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(24, 3, padding='same', activation=tf.nn.relu)(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(padding='same')(conv2)
    conv3 = tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu)(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(padding='same')(conv3)
    pool3_flat = tf.keras.layers.Flatten()(pool3)
    #print(pool3_flat.shape)
    reshaped = tf.keras.layers.Reshape((1,800))(pool3_flat)
    #print(reshaped.shape)
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, input_shape=(reshaped.shape[1]*reshaped.shape[2],32),return_sequences=True))(reshaped)
    linear1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(lstm1)
    drop = tf.keras.layers.Dropout(rate=0.5)(linear1)
    soft = tf.keras.layers.Dense(4, activation=tf.nn.softmax)(drop)
    
    return tf.keras.Model(inputs=inputs, outputs=soft)

def create_cnn():
    model = tf.keras.models.Sequential()
    #layer1
    model.add(tf.keras.layers.Conv1D(16, 3, padding='same', input_shape=(384, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    #layer2
    model.add(tf.keras.layers.Conv1D(24, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    #layer3
    model.add(tf.keras.layers.Conv1D(32, 3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    
    #Flatten
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    #model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(4, kernel_initializer='glorot_uniform', activation='softmax')) # EMOTION NUMBERS
    return model

sess = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
for sp in sess:
    # DATASET FOR LSTM
    with gzip.open("/home/gnlenfn/remote/lstm/preprocess/spectrogram/norm_spec.pkl") as ifp:
        data = pickle.load(ifp)
        
        test = data[data['speaker'] == sp]
        test = test[test.type == 'impro'] # for only impro record
        t_neu = test[test.label == 'neu']
        t_ang = test[test.label == 'ang']
        t_sad = test[test.label == 'sad']
        t_hap = test[test.label == 'hap']
        t_exc = test[test.label == 'exc']
        # hap + exc
        hap_t = pd.concat([t_hap, t_exc])
        hap_t = hap_t.replace("exc", "hap").reset_index(drop=True)
        le = LabelEncoder()
        test_sep = pd.concat([t_neu, t_ang, t_sad, hap_t]).reset_index(drop=True)
        lstm_test = test_sep['feature']
        lstm_y = test_sep['label']
        lstm_y = le.fit_transform(lstm_y)

        lstm_test = tf.convert_to_tensor(lstm_test)
        #print(lstm_test.shape)
        lstm_test = tf.reshape(lstm_test,[lstm_test.shape[0], 1, 200, 300])
    
    # DATASET FOR CNN
    with gzip.open("/home/gnlenfn/remote/spec2img/python/IS09_emotion_feature.pkl", 'r') as ifp:
        raw_data = pickle.load(ifp)
    raw_data = raw_data[raw_data.Type == 'impro']
    neu = raw_data[raw_data.EMOTION == 'neu']
    ang = raw_data[raw_data.EMOTION == 'ang']
    sad = raw_data[raw_data.EMOTION == 'sad']
    hap = raw_data[raw_data.EMOTION == 'hap']
    exc = raw_data[raw_data.EMOTION == 'exc']
    hap_m = pd.concat([hap, exc]).replace("exc", "hap").reset_index(drop=True) # Merge happy and excited
    df = pd.concat([neu, ang, hap_m, sad]).reset_index(drop=True) # Check EMOTION NUMBERS in MODEL
    df['name'] = df['name'].apply(stp)
    
    testdf = df[df['Session'] == sp].reset_index(drop=True)
    test_imb = testdf.iloc[:,1:-3].reset_index(drop=True)
    test_label = testdf['EMOTION']
    
    #cnn_test_over, cnn_y_over = SMOTE(random_state=4).fit_resample(test_imb, test_label)
    cnn_test = np.array(test_imb).reshape(test_imb.shape[0], 384, 1)
    lb = LabelEncoder()
    cnn_y = to_categorical(lb.fit_transform(test_label))
    print(lstm_test.shape, cnn_test.shape)
    model_lstm = build_lstm()
    model_cnn = create_cnn()
    
    lstm_list = glob.glob("/home/gnlenfn/remote/lstm/model/4emo/20hz/" + sp + "/*.h5")
    model_lstm.load_weights(lstm_list[-1])
    
    cnn_list = glob.glob("/home/gnlenfn/remote/spec2img/model/384dim/4emo/oversample/v2/" + sp + "/*.h5")
    model_cnn.load_weights(cnn_list[-2])
    
    models = [model_lstm, model_cnn]
    
    
    outputs = [models[0].predict(lstm_test), models[1].predict(cnn_test)]
    y = tf.keras.layers.Average()(outputs)
    #summed = np.sum(outputs, axis=1)
    #result = np.argmax(summed, axis=1)
    print(outputs)

