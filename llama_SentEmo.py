
#For HPC Workshop
from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
#from transformers import LlamaTokenizer
import transformers
import torch
import os
import pandas as pd
import re

from tqdm import tqdm

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

access_token = "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" #Pranay's access token, please don't abuse it :(

model = "BramVanroy/Llama-2-13b-chat-dutch"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=access_token)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    #use_auth_token=access_token
)

def get_sentiment3(text_piece):
    prompt = f'Welke sentiment wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze drie opties: "positief", "negatief", "neutraal".\nAntwoord: '
    #prompt = f'Welke gevoel wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze drie opties: "positief", "negatief", "neutraal".\nAntwoord: '
    #prompt = f'Welke sentiment wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze vijf opties: "positief", "heel positief", "negatief", "heel negatief", "neutraal".\nAntwoord: '
    #prompt = f'Welke gevoel wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze vijf opties: "positief", "heel positief", "negatief", "heel negatief", "neutraal".\nAntwoord: '
    full_prompt = """[INST] <<SYS>>
Je bent een behulpzame, respectvolle en eerlijke assistent. Antwoord altijd zo behulpzaam mogelijk. Je antwoorden mogen geen schadelijke, onethische, racistische, seksistische, gevaarlijke of illegale inhoud bevatten. Zorg ervoor dat je antwoorden sociaal onbevooroordeeld en positief van aard zijn.

Als een vraag nergens op slaat of feitelijk niet coherent is, leg dan uit waarom in plaats van iets niet correct te antwoorden. Als je het antwoord op een vraag niet weet, deel dan geen onjuiste informatie.
<</SYS>>

{prompt} [/INST] """

   
    #full_prompt = '''[INST] <<SYS>>
    #You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    #<</SYS>>
    
    #''' + prompt + '[/INST]'
    sequences = generator(
        prompt, # or full_prompt: this gave worse results for Aaron and Luna so first try with prompt
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=20,
    )

    for seq in sequences:
        response = seq['generated_text'].split(prompt)[1]
        if "neutraal" in response.lower():
            pred = 0
        elif "positief" in response.lower():
            pred = 1
        elif "negatief" in response.lower():
            pred = 2
        else:
            pred = "NA"

    return response, pred

def get_sentiment5(text_piece):
    # prompt = f'Welke sentiment wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze drie opties: "positief", "negatief", "neutraal".\nAntwoord: '
    #prompt = f'Welke gevoel wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze drie opties: "positief", "negatief", "neutraal".\nAntwoord: '
    prompt = f'Welke sentiment wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze vijf opties: "positief", "heel positief", "negatief", "heel negatief", "neutraal".\nAntwoord: '
    #prompt = f'Welke gevoel wordt in deze tekst uitgedrukt? Tekst: "' + text_piece + '". Antwoord met één van deze vijf opties: "positief", "heel positief", "negatief", "heel negatief", "neutraal".\nAntwoord: '
    full_prompt = """[INST] <<SYS>>
    Je bent een behulpzame, respectvolle en eerlijke assistent. Antwoord altijd zo behulpzaam mogelijk. Je antwoorden mogen geen schadelijke, onethische, racistische, seksistische, gevaarlijke of illegale inhoud bevatten. Zorg ervoor dat je antwoorden socia>

    Als een vraag nergens op slaat of feitelijk niet coherent is, leg dan uit waarom in plaats van iets niet correct te antwoorden. Als je het antwoord op een vraag niet weet, deel dan geen onjuiste informatie.
    <</SYS>>

    {} [/INST] """.format(prompt)


    #full_prompt = '''[INST] <<SYS>>
    #You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    #<</SYS>>
    #''' + prompt + '[/INST]'
    
    sequences = generator(
        full_prompt, #or full_prompt: this gave worse results for Aaron and Luna so first try with prompt
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=20,
    )

    for seq in sequences:
        #print(seq['generated_text'])
        response = seq['generated_text'].split(prompt)[1]
        if "neutraal" in response.lower():
            pred = "Neutraal"
        elif "heel positief" in response.lower():
            pred = "Zeer Positief"
        elif "heel negatief" in response.lower():
            pred = "Zeer Negatief"
        elif "positief" in response.lower():
            pred = "Positief"
        elif "negatief" in response.lower():
            pred = "Negatief"
        else:
            pred = "Neutraal"

    return response, pred

from transformers import pipeline
import pandas as pd
import ast
import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
import torch
from prompt import Prompting

def get_instance(object):
    sent_mapping = {'neg': 'Negatief', 'very_neg': 'Zeer Negatief', 'pos' : 'Positief', 'very_pos' : 'Zeer Positief', 'neu' : "Neutraal"}
    indices = object['aspect_index'].replace('[','').replace(']','').split(',')
    sentence = object['sentence']
    tokens = ast.literal_eval(object['token'])
    aspect = ''
    for index in indices:
        aspect += tokens[int(index)] + ' '
    aspect = aspect.strip()
    sent = object['sentiment']
    return sentence, aspect, sent_mapping[sent]
    

eval_file = 'data/hotel_nl_abea.csv'
df = pd.read_csv(eval_file, delimiter=';')
df.dropna(inplace=True)

true_labels = []
predicted_labels = []

for index, row in tqdm.tqdm(df.iterrows()):
    text_piece, aspect, true = get_instance(row)
    labels = ["Zeer Negatief", "Negatief", "Neutraal", "Positief",  "Zeer Positief"]
    true_labels.append(true)
    response, pred = get_sentiment5(text_piece)
    predicted_labels.append(pred)

print(classification_report(true_labels, predicted_labels))



