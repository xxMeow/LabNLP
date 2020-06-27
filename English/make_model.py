# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import json
import csv
import os  # To access local files
import re
import pandas as pd
import nltk
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Flatten, Dropout
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

nltk.download('stopwords')


def printDialogs(dialogs):
    for dialog in dialogs:
        for speaking in dialog:
            for key in speaking:
                print(key + ":\t" + speaking[key])
            print()
        print("-------------------------------------------------------------------------")


def generateNewJson(jsonFileName, dialogs):  # Generate as newJsonFile with clean data
    with open(jsonFileName + "_new.json", "w") as newJsonFile:
        json.dump(dialogs, newJsonFile, indent=4, separators=(',', ': '))


def loadDialogs(jsonFileName):
    # https://www.fileformat.info/info/unicode/char/2014/index.htm
    removeMap = {  # 위 링크를 참고해서 인코딩을 mapping함
        "\x85": "…",
        "\x91": "'",
        "\x92": "'",
        "\x93": "\"",
        "\x94": "\"",
        "\x96": "-",
        "\x97": "-",
        "\xa0": "",
        "\xe8": "e",
        "\xe9": "e",
        "\u2014": "-",
        "\u2019": "'",
        "\u2026": "…"
    }

    # Read JSON
    with open(jsonFileName + ".json") as jsonFile:
        dialogs = json.load(jsonFile)

    # Modify invalid encoding and remove redundant spaces
    pattern = re.compile(r"\s+")
    for dialog in dialogs:
        for speaking in dialog:
            for key in removeMap:  # invalid한 인코딩을 바꿔줌
                speaking["utterance"] = speaking["utterance"].replace(
                    key, removeMap[key])
            # 공백 제거 & lowercase로 변환
            speaking["utterance"] = (
                re.sub(pattern, " ", speaking["utterance"])).lower()
            # speaking["utterance"] = speaking["utterance"].replace(
            #    "[^a-z ]", "")
            speaking["utterance"] = re.sub(
                "[^a-z ]", "", speaking["utterance"])

    return dialogs


def tokenizeDialogs(dialogs):
    friendsStopwords = stopwords.words("english")
    for dialog in dialogs:
        for speaking in dialog:
            # print(speaking["utterance"])  # 데이터를 확인하기 위해 command line에다 출려함
            speaking["utterance"] = nltk.regexp_tokenize(
                speaking["utterance"], "[\w']+")  # nltk와 정규표현식을 이용하여 토큰화
            speaking["utterance"] = [
                word for word in speaking['utterance'] if word not in friendsStopwords]
            # print(speaking["utterance"])  # 데이터를 확인하기 위해 command line에다 출려함
    return dialogs

    # Process real data!!
path = os.getcwd() + "/"
jsonFileNames = ["friends_train", "friends_dev", "friends_test"]
for jsonFileName in jsonFileNames:
    dialogs = loadDialogs(path + jsonFileName)
    tokenizeDialogs(dialogs)
    # 새 JSON파일 이름은 원래 이름 뒤에다가 "_new"를 붙여서 generate됨
    generateNewJson(path + jsonFileName, dialogs)
    # 처리된 데이터들은 friends_xxx_new.json 파일에서 확인가능

# preprocess된 json파일로부터 utterance와 annotation을 읽어옵니다


def loadUtteranceSet(jsonFileName):  # Load clean dialogs from JSON
    with open(jsonFileName + ".json") as jsonFile:
        dialogs = json.load(jsonFile)  # Read JSON

        utteranceSet = []  # ONE utteranceSet for ONE file (dialog별로 구분되지 않습니다)
        for dialog in dialogs:
            for speaking in dialog:
                utterance = []
                utterance.append(speaking["utterance"])
#                utterance.append(speaking["annotation"])
                utterance.append(speaking["emotion"])
# 결과가 이상해서 임시로 emotion으로 해 보았습니다.
                utteranceSet.append(utterance)
    return utteranceSet


def makeDataFrame(utteranceSet):
    # 두 column의 이름을 지정해주며 dataframe을 생성합니다
    frame = pd.DataFrame(utteranceSet, columns=["utterance", "annotation"])
    return frame


path = os.getcwd() + "/"
jsonFileNames = ["friends_dev_new", "friends_test_new",
                 "friends_train_new"]  # 읽고자 하는 파일 이름을 확장자 없이 여기에 넣으시면 됩니다

frames = []  # ONE frame <- ONE utteranceSet <- ONE file
for jsonFileName in jsonFileNames:
    utteranceSet = loadUtteranceSet(path + jsonFileName)
    # pandas로 frame 하나를 생성해서 리스트에 추가
    frames.append(makeDataFrame(utteranceSet))
'''
for i in range(len(jsonFileNames)):  # 앞서 만든 dataframe들을 하나씩 출력해봅니다
    print("< " + jsonFileNames[i] +
          " > -------------------------------------------------")
    print(frames[i])
    print()
'''
# printTopUtterances(utterances, 5)

dev_data = frames[0]
test_data = frames[1]
train_data = frames[2]
X_dev = dev_data["utterance"].tolist()
X_test = test_data["utterance"].tolist()
X_train = train_data["utterance"].tolist()


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
vocab_size = total_cnt - rare_cnt + 1

tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_dev = tokenizer.texts_to_sequences(X_dev)
y_dev = np.array(dev_data['annotation'])
y_train = np.array(train_data['annotation'])
y_test = np.array(test_data['annotation'])


# 빈 샘플 제거
drop_train = [index for index, sentence in enumerate(
    X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(
    X_test) if len(sentence) < 1]
drop_dev = [index for index, sentence in enumerate(
    X_dev) if len(sentence) < 1]

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

X_test = np.delete(X_test, drop_test, axis=0)
y_test = np.delete(y_test, drop_test, axis=0)

X_dev = np.delete(X_dev, drop_dev, axis=0)
y_dev = np.delete(y_dev, drop_dev, axis=0)

# 패딩


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1


max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
X_dev = pad_sequences(X_dev, maxlen=max_len)


def trans_y(y):
    emotionMap = {
        'neutral': 0,
        'joy': 1,
        'sadness': 2,
        'fear': 3,
        'anger': 4,
        'surprise': 5,
        'disgust': 6,
        'non-neutral': 7
    }
    temp = []
    for i in y:
        temp.append(emotionMap.get(i, 'error'))
    return temp


y_dev = np.array(trans_y(y_dev))
y_test = np.array(trans_y(y_test))
y_train = np.array(trans_y(y_train))
'''
print(len(X_dev))
print(len(y_dev))
print(len(X_test))
print(len(y_test))
print(len(X_train))
print(len(y_train))
'''


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


# 순환 신경망 모델 생성
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(8, activation="softmax"))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model_1.h5', monitor='val_acc',
                     mode='max', verbose=1, save_best_only=True)
model_json = model.to_json()
with open("best_model_1.json", "w") as json_file:
    json_file.write(model_json)

model.compile(optimizer='rmsprop', loss=f1_loss, metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[
                    es, mc], batch_size=60, validation_split=0.1)


# 컨볼루션 신경망 모델 생성
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model_2.h5', monitor='val_acc',
                     mode='max', verbose=1, save_best_only=True)
model_json = model.to_json()
with open("best_model_2.json", "w") as json_file:
    json_file.write(model_json)

model.compile(optimizer='rmsprop', loss=f1_loss, metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[
                    es, mc], batch_size=60, validation_split=0.1)


# 순환 컨볼루션 신경망 모델
model = Sequential()
model.add(Embedding(20000, 128, input_length=max_len))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(8, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model_3.h5', monitor='val_acc',
                     mode='max', verbose=1, save_best_only=True)
model_json = model.to_json()
with open("best_model_3.json", "w") as json_file:
    json_file.write(model_json)

model.compile(optimizer='rmsprop', loss=f1_loss, metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[
                    es, mc], batch_size=60, validation_split=0.1)
