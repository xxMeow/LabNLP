# -*- coding: utf-8 -*-

import json # To process JSON file
import os # To access local files

# print the top several(= num) utterances
def printTopUtterances(utterances, num):
    for i in range(num):
        print(str(utterances[i]))
    print("-----------------------------------")

# Read only utterances and their annotations from new JSON file
def loadUtterances(jsonFileName): # Load clean dialogs from JSON
    # Read JSON
    with open(jsonFileName + ".json") as jsonFile:
        dialogs = json.load(jsonFile)

        utterances = []
        for dialog in dialogs:
            for speaking in dialog:
                utterance = []
                utterance.append(speaking["utterance"])
                utterance.append(speaking["annotation"])
                utterances.append(utterance)
    return utterances

# Test
path = os.getcwd() + "/"
jsonFileName = "friends_dev_new" # 읽고자 하는 파일 이름을 확장자 없이 여기에 넣으시면 됩니다
utterances = loadUtterances(path + jsonFileName)
printTopUtterances(utterances, 5)