'''
import Libraries
'''

import pandas as pd
import urllib.request
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Flatten, Dropout
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.datasets import imdb
import tensorflow as tf
import keras.backend as K
import os


'''
Preprocessing
'''

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

# 변수에 데이터 저장
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

train_data = train_data.dropna()    # NaN값 제거

train_data['document'] = train_data['document'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 한글과 공백을 제외하고 모두 제거

train_data['document'].replace('', np.nan, inplace=True)  # 빈칸을 NaN값으로 변환
train_data = train_data.dropna(how='any')   # NaN값 제거

# 같은 방식으로 테스트 데이터 전처리
test_data['document'] = test_data['document'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거

'''
데이터 토큰화
'''

# 토큰화
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘',
             '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
our_stopwords = ['을', '이다', '다', '로', '점', '에서', '것', '내', '그',
                 '나', '안', '인', '게', '고', '아', '거', '요', '중', '네', '수', '저', '걸']
stopwords = stopwords + our_stopwords
okt = Okt()
X_train = []
for sentence in train_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_train.append(temp_X)
X_test = []
for sentence in test_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_test.append(temp_X)

'''
데이터 패딩 및 정수화
'''

# 정수 인코딩
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

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])


# 빈 샘플 제거
drop_train = [index for index, sentence in enumerate(
    X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(
    X_test) if len(sentence) < 1]

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

X_test = np.delete(X_test, drop_test, axis=0)
y_test = np.delete(y_test, drop_test, axis=0)


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


'''
모델 생성
'''

# f1 score 계산 함수


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
model.add(Dense(1, activation='sigmoid'))

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
model.add(Dense(1, activation='sigmoid'))

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
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model_3.h5', monitor='val_acc',
                     mode='max', verbose=1, save_best_only=True)
model_json = model.to_json()
with open("best_model_3.json", "w") as json_file:
    json_file.write(model_json)

model.compile(optimizer='rmsprop', loss=f1_loss, metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[
                    es, mc], batch_size=60, validation_split=0.1)


'''
모델 정확도 측정
'''

json_file = open("best_model_1.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model_1.h5")
loaded_model.compile(optimizer='rmsprop',
                     loss=f1_loss, metrics=['acc'])
print("\n 순환 신경망 모델의 테스트 정확도: %.4f" %
      (loaded_model.evaluate(X_test, y_test)[1]))

json_file = open("best_model_2.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model_2.h5")
loaded_model.compile(optimizer='rmsprop',
                     loss=f1_loss, metrics=['acc'])
print("\n 컨볼루션 신경망 모델의 테스트 정확도: %.4f" %
      (loaded_model.evaluate(X_test, y_test)[1]))

json_file = open("best_model_3.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model_3.h5")
loaded_model.compile(optimizer='rmsprop',
                     loss=f1_loss, metrics=['acc'])
print("\n 순환 컨볼루션 신경망 모델의 테스트 정확도: %.4f" %
      (loaded_model.evaluate(X_test, y_test)[1]))


'''
모델을 사용하고 싶을 때는 아래쪽 코드의 주석을 해제
'''

'''
# 이 윗줄을 지워주세요

# 1. 실무에 사용할 데이터 준비하기
test_data = pd.read_csv('data/ko_data.csv', encoding='CP949')
# print(test_data[:5])
test_data['Sentence'] = test_data['Sentence'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['Sentence'].replace(np.nan, '', inplace=True)
okt = Okt()
X_test = []
for sentence in test_data['Sentence']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    X_test.append(temp_X)


X_test = tokenizer.texts_to_sequences(X_test)

test_data['Sentence'].replace(np.nan, '', inplace=True)

# 패딩


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1


max_len = 30
X_test = pad_sequences(X_test, maxlen=max_len)


# 2. 모델 불러오기
json_file = open("best_model_3.json", "r")  # 적용을 원하는 모델의 json파일
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model_3.h5")  # 적용을 원하는 모델의 h5파일
loaded_model.compile(optimizer='rmsprop',
                     loss=f1_loss, metrics=['acc'])

# 3. 모델 사용하기
result = loaded_model.predict_classes(X_test)

# 4. 결과 저장
loaded_result = pd.read_csv('data/ko_sample.csv')
loaded_result["Predicted"] = result
print(loaded_result)
loaded_result.to_csv('data/ko_result_3.csv', index=None)  # 결과가 저장될 파일 이름

# 이 아래쪽을 지워주세요
'''
