import Update_model_for_chatbot
import regex
import json
import requests

api_key = "7c82a033-b166-4a75-827d-1cbb79f73f8b"
url = "https://api.oneai.com/api/v0/pipeline"


questions = Update_model_for_chatbot.question
answers = Update_model_for_chatbot.answer
possible_yes = ['y', 'Y', 'yes', 'Yes']
possible_no = ['N', 'n', 'no', 'No']


def bingbing():
    Qinp = input("What would you like to search today?: \n")
    tokens = Qinp.split()
    copy_tokens = tokens.copy()
    for x in copy_tokens:
        if x in "HOWHowhowWHATwhatISisAREareTHEtheANDandDOESdoesITitOFof":
            tokens.remove(x)
        if x.endswith("?"):
            tokens.remove(x)
            x = x.replace("?","")
            tokens.append(x)
#    print(tokens)


    response = []
    c = 1
    for i in tokens:
        for x in answers:
            if i.lower() in x or i.upper() in x:
                response.append(x)
                answers.remove(x)
                break
        c+=1
        if c >= len(tokens):
            break
    for i in response:
        print(i)

    headers = {
        "api-key": api_key,
        "content-type": "application/json"
    }
    payload = {
        "input": str(response),
        "input_type": "article",
        "output_type": "json",
        "multilingual": {
            "enabled": True
        },
        "steps": [
            {
                "skill": "summarize"
            },
            {
                "skill": "content-curation"
            }
        ],
    }

    r = requests.post(url, json=payload, headers=headers)
    data = r.json()

bingbing()
