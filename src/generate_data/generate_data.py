import pandas as pd
import re
import string
import torch
import spacy
import numpy as np
from tqdm import tqdm

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline
from generate_data.core_translate import create_translate_model, load_spacy_model, translate

device = "cuda" if torch.cuda.is_available() else "cpu"



DCT_SPECIAL_WORD = {
    "<adjective>",
    "<budget>",
    "<confirmation_code>",
    "<country>",
    "<date>",
    "<duration>",
    "<email>",
    "<hotel>",
    "<keyword>",
    "<language>",
    "<location>",
    "<lunch_or_dinner>",
    "<month>",
    "<name>",
    "<number_of_people>",
    "<restaurant>",
    "<time>",
    "<tour>",
    "<tour_parameter>",
    "<transportation>"
}
"""
DCT_SPECIAL_WORD_REPLACE = {
    "<adjective>",
    "<budget>",
    "<confirmation_code>": ["2345", "123", "],
    "<country>": ["France", "USA", "England", "Thailand", "China", "Vietnam"],
    "<date>": ["2011-02-18", "2014-08-13", "2017-02-08", "2012-02-29", "2015-05-23", "30, Jun 2009", "07, Oct 2012", "28, Feb 2014", "10, Dec 2016", "28, Aug 1999"],
    "<duration>": ["in 2 hours", "a century", "10 years", ],
    "<email>": ["james@yahoo.com", "robert@gmail.com", "john@gmail.com", "michael@yahoo.com", "william@gmail.com", "mary@gmail.com", "patricia@yahoo.com", "jennifer@gmail.com", "linda@gmail.com", "elizabeth@yahoo.com"],
    "<hotel>": ["Capella Ubud", "Hotel Amparo", "Fogo Island Inn", "The Ritz-Carlton", "Waldorf Astoria Maldives Ithaafushi", "Secret Bay", "Raffles Istanbul", "Canaves Oia Epitome", "Awasi Patagonia"],
    "<keyword>",
    "<language>": ["English", "Mandarin Chinese", "Hindi", "Spanish", "French", "Standard Arabic", "Bengali", "Russian", "Portuguese", "Indonesian"],
    "<location>" : ["my house", "my school", "market", "Paris", "New York", "London", "Bangkok", "Dubai", "Rome"],
    "<lunch_or_dinner>": ["lunch", "dinner"],
    "<month>": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "<name>": ["James", "Robert", "John", "Michael", "William", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth"],
    "<number_of_people>": ["1", "3", "4", "7", "9", "20"],
    "<restaurant>": ["Mirazur", "Noma", "Asador Etxebarri", "Gaggan", "Geranium", "Central", "Mugaritz", "Arpège"],
    "<time>": ["tomorrow", "7 pm", "7:30 tonight", "yesterday", "10 am"],
    "<tour>",
    "<tour_parameter>",
    "<transportation>"
}
"""

##### PARAPHRASE SIMILAR
def load_model_paraphrase(model_name="tuner007/pegasus_paraphrase"):
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer

def text_augmentation_by_paraphrase(input_text, model, tokenizer, num_return_sequences=5, num_beams=10):
    batch = tokenizer([input_text],truncation=True,padding="longest",max_length=60, return_tensors="pt").to(device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return tgt_text


#### FILL MASK
def load_model_fill_mask():
    model = pipeline("fill-mask")
    return model

def text_augmentation_by_fill_mask(text, model, spacy_nlp, threshold=0.2):
    # choose words to make a mask
    def choose_word(text, nlp):
        # remove <>
        text = text.translate(str.maketrans("", "", """.,/?!#@$%^&*()-+=`~"""))
        text_split = text.split(" ")
        text = " ".join([w for w in text_split if w not in DCT_SPECIAL_WORD])

        doc = nlp(text)
        words = []
        for token in doc:
            if token.pos_ in {"VERB", "PROPN", "NOUN", "ADJ"} and text_split.count(token.text) == 1:
                words.append(token.text)
        return words
    
    select_words = choose_word(text, spacy_nlp)
    
    result = []
    if len(select_words) > 0:
        choice_word = np.random.choice(select_words, size=min(len(select_words), 5), replace=False)
        # replace words by <mask>
        for word in choice_word:
            try:
                result += model(text.replace(word, "<mask>"))
            except:
                pass
    result = [r["sequence"] for r in result if r["score"] > threshold]
    return result


##### TRANSLATION
def load_model_translate(lang_intermedia="German"):
    model_english_intermedia = create_translate_model(src_lang="English", target_lang=lang_intermedia)
    model_intermedia_english = create_translate_model(src_lang=lang_intermedia, target_lang="English")

    spacy_intermedia = load_spacy_model(lang_intermedia)
    spacy_english = load_spacy_model("English")

    return model_english_intermedia, spacy_intermedia, model_intermedia_english, spacy_english

def text_augmentation_by_translation(text, model_english_intermedia, spacy_intermedia, model_intermedia_english, spacy_english):
    text_trans = translate(model_english_intermedia, spacy_intermedia, text)
    text_reverse_trans = translate(model_intermedia_english, spacy_english, text_trans)
    return text_reverse_trans




# Load model
print("Load models for text agumenttation")
model_paraphrase, tokenizer = load_model_paraphrase()
    
nlp = spacy.load("en_core_web_sm")
model_fill_mask = load_model_fill_mask()

model_english_german, spacy_german, model_german_english, spacy_english = load_model_translate(lang_intermedia="German")
model_english_french, spacy_french, model_french_english, spacy_english = load_model_translate(lang_intermedia="French")

def generate_more_data(df):

    def text_augmentation(text, by="paraphrase"):
        if by == "paraphrase":
            return text_augmentation_by_paraphrase(text, model_paraphrase, tokenizer)
        elif by == "fill mask":
            return text_augmentation_by_fill_mask(text, model_fill_mask, nlp)
        elif by == "translate":
            text_ger_trans = text_augmentation_by_translation(text, model_english_german, spacy_german, model_german_english, spacy_english)
            text_fren_trans = text_augmentation_by_translation(text, model_english_french, spacy_french, model_french_english, spacy_english)
            return [text_ger_trans, text_fren_trans]


    # df = pd.read_csv("question.csv")
    bar = tqdm(df.iterrows(), total=df.shape[0], desc="Generate more data...")
    data = []
    for i,row in bar:
        utterance, intent = row["Utterance"], row["Intent"]
        for i,sw in enumerate(DCT_SPECIAL_WORD):
            code = str((i+1)*13)
            code = string.ascii_lowercase[i].upper() + '0'*(3-len(code))+code
            utterance = utterance.replace(sw, code)

        lst_para_lv1 = text_augmentation(utterance, by="paraphrase")
        #lst_fill_lv1 = text_augmentation(utterance, by="fill mask")
        lst_trans_lv1 = text_augmentation(utterance, by="translate")

        #lst_para_fill_lv2 = []
        #for text in lst_para_lv1:
        #    lst_para_fill_lv2 += text_augmentation(text, by="fill mask")

        #lst_fill_trans_lv2 = []
        #for text in lst_fill_lv1:
        #    lst_fill_trans_lv2 += text_augmentation(text, by="translate")

        # lst_trans_para_lv2 = []
        # for text in lst_trans_lv1:
        #     lst_trans_para_lv2 += text_augmentation(text, by="paraphrase")

        #all_text = list(set(lst_para_lv1 + lst_fill_lv1 + lst_trans_lv1 + lst_para_fill_lv2 + lst_fill_trans_lv2 + lst_trans_para_lv2)) + [utterance]
        all_text = list(set(lst_para_lv1 + lst_trans_lv1)) + [utterance]

        for idt,text in enumerate(all_text):
            for i,sw in enumerate(DCT_SPECIAL_WORD):
                code = str((i+1)*13)
                code1 = string.ascii_lowercase[i].upper() + '0'*(3-len(code))+code
                code2 = string.ascii_lowercase[i].upper() + ' '+ '0'*(3-len(code))+code
                
                code = str((i+1)*13+1)
                code3 = string.ascii_lowercase[i].upper() + ' '+ '0'*(3-len(code))+code
                text = text.replace(code1, sw)
                text = text.replace(code2, sw)
                text = text.replace(code3, sw)

            # random value for special phrase <>
            sample = {
                "Utterance": text,
                "Intent": intent
            }
            data.append(sample)
            
    df_generate = pd.DataFrame(data)
    
    return df_generate

# if __name__ == "__main__":
#     main()



"""
def find_all_substring_between_subs(text, first, last):
    subs = []
    index = 0
    # res = re.search(f"{first}(.*?){last}", text[index:])
    while index < len(text):
        res = re.search(f"{first}(.*?){last}", text[index:])
        if res is not None:
            subs.append(res.group(1))
            index += res.span()[1]
        else:
            break
    return subs

df = pd.read_csv("question.csv")
all_special_word = []
for i,row in df.iterrows():
    all_special_word += find_all_substring_between_subs(row["Utterance"], "<", ">")
all_special_word = set(all_special_word)
"""
