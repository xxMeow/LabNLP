import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import model_from_json
import os

# 1. 실무에 사용할 데이터 준비하기
data_path = (os.path.dirname(os.path.realpath(__file__)))+'/data/ko_data.csv'
test_data = pd.read_csv(data_path, encoding='CP949')
# print(test_data[:5])
test_data['Sentence'] = test_data['Sentence'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['Sentence'].replace(np.nan, '', inplace=True)
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘',
             '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
our_stopwords = ['을', '이다', '다', '로', '점', '에서', '것', '내', '그',
                 '나', '안', '인', '게', '고', '아', '거', '요', '중', '네', '수', '저', '걸']
stopwords = stopwords + our_stopwords
okt = Okt()
X_test = []
for sentence in test_data['Sentence']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_test.append(temp_X)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_test)

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
tokenizer.fit_on_texts(X_test)
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
json_file = open("best_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model.h5")
loaded_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy', metrics=['acc'])

# 3. 모델 사용하기
result = loaded_model.predict_classes(X_test)

# print(result[:5])
# print(len(result))

# 4. 결과 저장
data_path = (os.path.dirname(os.path.realpath(__file__)))+'/data/ko_sample.csv'
loaded_result = pd.read_csv(data_path)
loaded_result["Predicted"] = result
# print(loaded_result)
data_path = (os.path.dirname(os.path.realpath(__file__)))+'/data/ko_result.csv'
loaded_result.to_csv(data_path, index=None)
