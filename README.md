### Table of Content



- [Question Augmentation and Intent Classification](#question-augmentation-and-intent-classification)
    - [Installation](#installation)
      - [Requirements](#requirements)
    - [Overview](#overview)
      - [Question Augmentation:](#question-augmentation)
        - [_Paraphrase_:](#paraphrase)
        - [_Back translate_:](#back-translate)
        - [_(Optional) Fill-Mask_:](#fill-mask)
        - [_Advanced Technique: GPT-3_](#advanced-technique-gpt-3)
      - [Intent Classification:](#intent-classification)
    - [Training And Evaluation:](#training-and-evaluation)
      - [Generation Question](#generation-question)
      - [Train Intent Classification](#train-intent-classification)
      - [Evaluation](#evaluation)



# Question Augmentation and Intent Classification


| ![architecture.png](https://user-images.githubusercontent.com/28798474/124345446-1cbb9480-dc03-11eb-96e7-edf4d76d2c60.png) |
|:--:|
| *Overview the Approach*|

### Installation
#### Requirements
```sh
pip install -r requirements.txt
```
#### 

### Overview
#### Question Augmentation: 
To generate more questions from original questions we use few augmentation techniques for text: Pharaphrase, Back-translation and Fill mask. 

##### _Paraphrase_:

I used Pharaphrase model inspired from Seq2seq model to paraphrase a question. I used HuggingFace public model for Pharaphrase from (
pegasus_paraphrase)[https://huggingface.co/tuner007/pegasus_paraphrase].

- For example: 
```
Original Question: Can you reset my password?
Paraphrased Question: Is it possible to reset my password?
```

- Code usage:
```python
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
```

##### _Back translate_:
This is quite common technique for Text Augmentation. From original question we will translate it into another language (e.g. France, German, Vietnamese, ...) and translate back to English
- Example: 
```
Original Question: I want to know the process little more careful and wonder if there are the backup for the payment if the reservation was a scam
German Question: Ich möchte den Prozess etwas vorsichtiger wissen und mich fragen, ob es die Sicherung für die Zahlung gibt, wenn die Reservierung ein Betrug war
Back translate: I want to know the process a little more carefully and ask myself if there is the backup for the charge if the reservation was a fraud
```

##### _(Optional) Fill-Mask_:
Transformer models such as BERT, ROBERTA, and ALBERT have been trained on a large amount of text using a pretext task called “Masked Language Modeling” where the model has to predict masked words based on the context.

This can be used to augment some text. But this cases we this technique still work not quite well so we just use this technique to augmente low ratio text.

- Example:
```
Original Question: I [MASK] my login information, can you help?
Fill Question: I need my login information, can you help?
```
- Code:
```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("I [MASK] my login information, can you help?")
```

##### _Advanced Technique: GPT-3_
I got permission access to OpenAI GPT-3. GPT-3 is a powerful and very popular model in recent times. I use GPT-3 to generate new questions based on the design promt. The prompt I already designed like this: 
```
[intent] [example_question_1]
[intent] [example_question_2]
[intent] [example_question_3]
```
- Example: 
```
Input: 
[tour_inquiry] Is lunch included in tour?
[tour_inquiry] What is the refund policy?
[tour_inquiry] What happens if I don't get the reservation on an order?
[tour_inquiry] Are kids allowed on this tour?

Generation Output:
[tour_inquiry] Can I cancel my booking?
[tour_inquiry] What is the cancellation policy?
[tour_inquiry] Can I change the booking?
[tour_inquiry] Can I make changes to my booking?
[tour_inquiry] What is the age limit for
```
- Code:
```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="davinci",
  prompt="[tour_inquiry] Is lunch included in tour?\n[tour_inquiry] What is the refund policy?\n[tour_inquiry] What happens if I don't get the reservation on an order?\n[tour_inquiry] Are kids allowed on this tour?\n[tour_inquiry] Can I cancel my booking?\n[tour_inquiry] What is the cancellation policy?\n[tour_inquiry] Can I change the booking?\n[tour_inquiry] Can I make changes to my booking?\n[tour_inquiry] What is the age limit for",
  temperature=0.7,
  max_tokens=64,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
"""Output
[tour_inquiry] Can I cancel my booking?
[tour_inquiry] What is the cancellation policy?
[tour_inquiry] Can I change the booking?
[tour_inquiry] Can I make changes to my booking?
[tour_inquiry] What is the age limit for this tour?
"""

```
Output GPT-3 at `data/gpt3_questions.txt` The problem is GPT-3 take a bit cost for API, I wil changed to use GPT-J.
#### Intent Classification:

When finished create data, I think intent classification is not hard problem. To solve classification problem, I used Bert Model, just noted that this problem is multibel text classification. We perform k-Fold CrossValidation. 

### Training And Evaluation:
For general training I split the original data into 5-folds. Use 4-folds as the train dataset and the other fold as the validation dataset.

#### Generation Question
```sh
cd src && python create_data_training.py
```

#### Train/CrossValidation Intent Classification
```sh
cd src && python train.py
```

#### Evaluation

* I will save the 5 best models corresponding to the 5-folds division above, the accuracy, confusion matrix of each model to choose the best model.
All outputs are stored in the output directory.
* Quick Result: Average accuracy on 5-Fold
    - Bert model on original data without augmentation: 54.34%
    - Bert model on generated data set: 61.5% 
#### Demo 
I already have trained models at this [google drive](https://drive.google.com/file/d/156IPrU6X-RsrJ-8h_ZK5PTYNIKO348zm/view?usp=sharing)
Download and copy to `output/` folder to run the demo.
```sh
cd src && python inference.py
```
