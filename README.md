# 자연어 처리 - 15조

## 1. 한국어 감정 분석

### 개발환경
python3

### 필요한 라이브러리

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
python3와 jdk, 필요한 라이브러리들이 모두 설치되어 있는지 확인하고 없을 시 설치할 것
#### 모델 생성
- 프로젝트를 clone <br>
``` git clone https://github.com/xxMeow/LabNLP.git ``` <br>
- 한글 폴더로 이동 <br>
``` cd Korean ``` <br>
- 모델의 실제 사용을 원할 시, kaggle에서 다운받은 ko_data.csv오 ko_sample.csv 파일을 data 폴더에 넣어둘 것
- 모델 생성 파일을 실행 <br>
``` python3 make_model.py ``` <br>

### 참고한 오픈소스

https://wikidocs.net/44249 <br>
https://tykimos.github.io/2017/06/10/Model_Save_Load/ <br>
https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric <br>
https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/

## 2. 영어 감정 분석

### 필요한 라이브러리

- json
- os
- re
- nltk
- pandas
- numpy
- tensorflow
- keras
- csv

### 실행방법

##### Google Colab으로 실행
- 프로젝트를 clone (이미 clone했다면 불필요)<br>
``` git clone https://github.com/xxMeow/LabNLP.git ``` <br>
- 영어 폴더로 이동 <br>
``` cd English ``` <br>
- 모델 생성 파일을 실행 <br>
``` python3 make_model.py ``` <br>

### 참고한 오픈소스

https://medium.com/appening-io/emotion-classification-2d4ed93bf4e2

