# -*- coding: utf-8 -*-

import json # To process JSON file
import os # To access local files
import re
import pandas as pd
import nltk # Natural Language ToolKit

def printDialogs(dialogs):
    for dialog in dialogs:
        for speaking in dialog:
            for key in speaking:
                print(key + ":\t" + speaking[key])
            print()
        print("-------------------------------------------------------------------------")

def generateNewJson(jsonFileName, dialogs): # Generate as newJsonFile with clean data
    with open(jsonFileName + "_new.json", "w") as newJsonFile:
        json.dump(dialogs, newJsonFile, indent=4, separators=(',', ': '))

def loadDialogs(jsonFileName):
    # https://www.fileformat.info/info/unicode/char/2014/index.htm
    removeMap = { # 위 링크를 참고해서 인코딩을 mapping함
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
            for key in removeMap: # invalid한 인코딩을 바꿔줌
                speaking["utterance"] = speaking["utterance"].replace(key, removeMap[key])
            speaking["utterance"] = (re.sub(pattern, " ", speaking["utterance"])).lower() ## 공백 제거 & lowercase로 변환
            
    return dialogs

def tokenizeDialogs(dialogs):
    pattern = r"""(?x)                 # set flag to allow verbose regexps 
	              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
	              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
	              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe 
	              |\.\.\.                # ellipsis 
	              |(?:[.,;"'?():-_`])    # special characters with meanings 
	            """

    for dialog in dialogs:
        for speaking in dialog:
            speaking["utterance"] = nltk.regexp_tokenize(speaking["utterance"], pattern) # nltk와 정규표현식을 이용하여 토큰화
            # print(speaking["utterance"]) # 데이터를 확인하기 위해 command line에다 출려함

    return dialogs

# Process real data!!
path = os.getcwd() + "/"
jsonFileNames = ["friends_train", "friends_dev", "friends_test"]
for jsonFileName in jsonFileNames:
    dialogs = loadDialogs(path + jsonFileName)
    tokenizeDialogs(dialogs)
    generateNewJson(path + jsonFileName, dialogs) # 새 JSON파일 이름은 원래 이름 뒤에다가 "_new"를 붙여서 generate됨
    # 처리된 데이터들은 friends_xxx_new.json 파일에서 확인가능