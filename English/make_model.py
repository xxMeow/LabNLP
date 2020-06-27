# -*- coding: utf-8 -*-

import json
import csv
import os  # To access local files
import pandas as pd
import nltk

# preprocess된 json파일로부터 utterance와 annotation을 읽어옵니다


def loadUtteranceSet(jsonFileName):  # Load clean dialogs from JSON
    with open(jsonFileName + ".json") as jsonFile:
        dialogs = json.load(jsonFile)  # Read JSON

        utteranceSet = []  # ONE utteranceSet for ONE file (dialog별로 구분되지 않습니다)
        for dialog in dialogs:
            for speaking in dialog:
                utterance = []
                utterance.append(speaking["utterance"])
                utterance.append(speaking["annotation"])
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

for i in range(len(jsonFileNames)):  # 앞서 만든 dataframe들을 하나씩 출력해봅니다
    print("< " + jsonFileNames[i] +
          " > -------------------------------------------------")
    print(frames[i])
    print()

# printTopUtterances(utterances, 5)
