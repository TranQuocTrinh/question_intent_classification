import spacy
from transformers import pipeline
import re
import unicodedata

def maptoalphanum(s):
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf8')


rec=re.compile(r'[^\x00-\x7f]')
rec2=re.compile('([.][ ]){2,}')
rec3=re.compile(r'\b(\w+)(?:\W+\1\b)+')
#recsplit=re.compile(r'\.|\!|\?|\:|\;|\-|\n')
recsplit=re.compile(r'(?<=\.)|(?<=\!)|(?<=\?)|(?<=\:)|(?<=\;)|(?<=\-)|(?<=\n)')

def latinonly(s):
    return rec.sub('',s)

def nodotspacerepeats(s):
    return rec2.sub('',s)

def remove_repeated_words(s):
    return rec3.sub('',s)

def create_translate_model(src_lang='German', target_lang='English'):
    list_code = ['German', 'English', 'French', 'Flemish', 'Swedish', 'Italian','Spanish']
    if src_lang not in list_code or target_lang not in list_code:
        raise Exception('Language not in setup list')

    dct_code = {
        "German": "de",
        "English": "en",
        "French": "fr",
        "Flemish": "nl",
        "Swedish": "sv",
        "Italian": "it",
        "Spanish": "es"
    }

    src_lang = dct_code[src_lang]
    target_lang = dct_code[target_lang]

    return pipeline("translation_xx_to_yy", model=f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}", device=0)


def load_spacy_model(src_lang):
    if src_lang == "German":
        nlp = spacy.load("de_core_news_sm")
    elif src_lang == "English":
        nlp = spacy.load("en_core_web_sm")
    elif src_lang == "French":
        nlp = spacy.load("fr_core_news_sm")
    elif src_lang == "Flemish":
        nlp = spacy.load("nl_core_news_sm")
    elif src_lang == "Swedish":
        nlp = spacy.load("sv_pipeline")
    else:
        nlp = spacy.load("xx_ent_wiki_sm")
    return nlp

def translate(model, spacy_nlp, text):


    text= remove_repeated_words(nodotspacerepeats(maptoalphanum(text)))
    #text = text.encode('ascii')
    CHUNK_SIZE = 150
    if isinstance(text,str)  and (len(text) <=5 or not ' ' in text):
        return text
    elif isinstance(text,str)  and len(text) >=6 and ' ' in text:
        doc = spacy_nlp(text)
        try:
            sents = [sent.text for sent in doc.sents]
        except:
            sents= [s.strip() for s in recsplit.split(text)]

        chunks = []
        sub_chunk = ""
        for (i, sent) in enumerate(sents):
            if len(sub_chunk.split(" ")) + len(sent.split(" ")) <= CHUNK_SIZE:
                sub_chunk = sub_chunk + " " + sent
                if i == len(sents) - 1:
                    chunks.append(sub_chunk)
            else:
                chunks.append(sub_chunk)
                sub_chunk = sent
        return ' '.join([x['translation_text'] for x in model(chunks, truncation='only_first')])
    else:
        return ''