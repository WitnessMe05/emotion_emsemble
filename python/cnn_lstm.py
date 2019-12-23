
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, gzip, os, glob
from numba import cuda
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            print("Directory is already exists")
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def build_model():
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
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, input_shape=(reshaped.shape[1]*reshaped.shape[2],32),return_sequences=True))(reshaped)
    linear1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(lstm1)
    drop = tf.keras.layers.Dropout(rate=0.5)(linear1)
    soft = tf.keras.layers.Dense(4, activation=tf.nn.softmax)(drop)
    
    return tf.keras.Model(inputs=inputs, outputs=soft)

batch_size = 64
input_dim = 25*32
units = 128
output_size = 4
epochs = 100

best_acc = []
sess = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
for sp in sess:
    with gzip.open("/home/gnlenfn/remote/lstm/preprocess/spectrogram/norm_spec.pkl") as ifp:
        data = pickle.load(ifp)

        # TRAINING SET
        train = data[data['speaker'] != sp]
        train = train[train.type == 'impro'] # for only impro record
        neu = train[train.label == 'neu']
        ang = train[train.label == 'ang']
        sad = train[train.label == 'sad']
        hap = train[train.label == 'hap']
        exc = train[train.label == 'exc']
        # hap + exc
        hap_m = pd.concat([hap, exc])
        hap_m = hap_m.replace("exc", "hap").reset_index(drop=True)
        emo_sep = pd.concat([neu, ang, sad, hap_m]).reset_index(drop=True)
        le = preprocessing.LabelEncoder()
        
        x_train = emo_sep['feature']
        y_train = emo_sep['label']
        y_train = le.fit_transform(y_train)
        # OVERSAMPLING
        #x_train, y_train = SMOTE(random_state=4).fit_resample(x_train, y_train)
        
        x_train = tf.convert_to_tensor(x_train)
        x_train = tf.reshape(x_train,[x_train.shape[0], 1, 200, 300])
        
        # TEST SET
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
        
        test_sep = pd.concat([t_neu, t_ang, t_sad, hap_t]).reset_index(drop=True)
        x_test = test_sep['feature']
        y_test = test_sep['label']
        y_test = le.fit_transform(y_test)
        # OVERSAMPLING
        #x_test, y_test = SMOTE(random_state=4).fit_resample(x_test, y_test)
        
        x_test = tf.convert_to_tensor(x_test)
        x_test = tf.reshape(x_test,[x_test.shape[0], 1, 200, 300])
        
        model_path = "/home/gnlenfn/remote/lstm/model/4emo/20hz/" + sp + "/"
        log_path = "/home/gnlenfn/remote/lstm/log/4emo/20hz/"
        createFolder(model_path)
        createFolder(log_path)
        
        model_name = model_path + sp + '-{epoch:03d}.h5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name, verbose=0, monitor='val_loss',
                                                        save_best_only=True, mode='auto')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
        
        with tf.device("/gpu:1"):
            model = build_model()
            model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(lr=0.0005), validation_data=(x_test, y_test),
                                metrics=['accuracy'])
            hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, 
                            validation_data=(x_test, y_test), callbacks=[checkpoint, tensorboard])
            
            loaded_model = build_model()
            tmp = glob.glob(model_path + "*")
            tmp.sort()
            print(tmp[-1])
            
            loaded_model.load_weights(tmp[-1])
            loaded_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                                metrics=['accuracy'])
            scores = loaded_model.evaluate(x_test, y_test, verbose=0)
            print("best model: {}: {:4f}%".format(loaded_model.metrics_names[1], scores[1]*100))
            best_acc.append(scores[1]*100)
            
            # CONFUSION MATRIX
            Y_pred = model.predict(x_test)
            pred=np.argmax(Y_pred.reshape(x_test.shape[0],4),axis=1)
            print(confusion_matrix(y_test, pred))
            target_names = ['ang', 'hap', 'neu', 'sad']
            print(classification_report(y_test, pred, target_names=target_names))
            
model.summary()
print("Total best avg accuracy: ", np.mean(best_acc))

cuda.select_device(1)
cuda.close()