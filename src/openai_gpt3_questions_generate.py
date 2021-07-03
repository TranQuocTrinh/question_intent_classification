# coding: utf-8
import pandas as pd
import random
import openai
from tqdm import tqdm
openai.api_key = 'sk-T8TuYGPTwhb8s4DobrQSAzinVVGhZtSd1ZnP8mXG'
df = pd.read_csv("questions.csv")
df.groupby('Intent')

dct_replace = {'<adjective>': 'beautiful',
        '<budget>' : "10000",
    '<confirmation_code>': "ABD102",
    '<country>': 'Vietnam',
    '<date>': '2011-02-18',
    '<duration>': '5 days',
    '<email>': 'james@yahoo.com',
    '<hotel>': 'Capella_Ubud',
    '<keyword>': "relax",
    '<language>': 'English',
    '<location>' : 'London',
    '<lunch_or_dinner>': 'lunch',
    '<month>': 'January',
    '<name>': 'James',
    '<number_of_people>': '3',
    '<restaurant>': 'Mirazur',
    '<time>': '7 pm',
    '<tour>': 'holiday',
    '<tour_parameter>': 'tour_parameter',
    '<transportation>': 'bus'
}


text_augmentation = {}
for key, df_sub in tqdm(df.groupby('Intent')):
    key_ = f"[{key}] "
    data = [key_ + x for x in df_sub['Utterance'].tolist() if "<" not in x]
    if len(data) == 0:
        for text in df_sub["Utterance"].tolist():
            for x in dct_replace:
                text = text.replace(x, dct_replace[x])
            data.append(key_ + text)
    prompt = "\n".join(data)
    prompt = prompt + "\n" + key_
    prompt = prompt.strip()
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      temperature=0.7,
      max_tokens=64,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    try:
        text_augmentation[key] = [x.replace("\n", "") for x in response["choices"][0]["text"].split(key_)[0:-1]]
    except:
        print("error: ", prompt)
import json
json.dump(text_augmentation, open("out.json", "w"))
