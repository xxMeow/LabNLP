# 자연어 처리 - 15조

## 1. 한국어 감정 분석

### 필요한 모듈

- pandas
- urllib
- matplotlib
- re
- konlpy
- tensorflow
- keras
- numpy
- os

### 실행방법
python3와 jdk, 필요한 모듈들이 모두 설치되어 있는지 확인하고 없을 시 설치할 것
#### 모델 생성
- 프로젝트를 clone <br>
``` git clone https://github.com/xxMeow/LabNLP.git ``` <br>
- 한글 폴더로 이동 <br>
``` cd Korean ``` <br>
- 모델 생성 파일을 실행 <br>
``` python3 make_model.py ``` <br>

### 참고한 오픈소스

https://wikidocs.net/44249 <br>
https://tykimos.github.io/2017/06/10/Model_Save_Load/ <br>
https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric <br>
https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/

## 2. 영어 감정 분석

### 필요한 모델

- python3
- jdk
- json
- os
- re
- nltk
- pandas
- numpy
- tensorflow
- keras

### 실행방법

##### Google Colab으로 실행

- [English.ipynb](https://github.com/xxMeow/LabNLP/blob/master/English/English.ipynb) 파일 다운로드
- 다운로드 받은 파일을 [Colab](https://colab.research.google.com/)에서 열기

- 윈쪽 파일 텝에 `friends_dev.json`, `friends_train.json`, `friends_test.json` 파일 추가
- 코드 실행

### 참고한 오픈소스

https://medium.com/appening-io/emotion-classification-2d4ed93bf4e2

