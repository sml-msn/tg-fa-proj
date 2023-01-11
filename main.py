import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from Levenshtein import distance, ratio

model_name = "sml-msn/pst5-tg-fa-bidirectional"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# transliteration function
def translit(x, **kwargs):
    inputs = tokenizer(x, return_tensors='pt').to(model.device)     
    with torch.no_grad():    
        hypotheses = model.generate(**inputs, **kwargs)        
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)

# checking if the input text has more than 500 symbols
# or it is empty
def check_length(txt):
	if len(txt) > 500:
		st.write('Output: ')
		st.write('Sorry, your text is too long.')
		return False
	elif txt == '':
		st.write('Output: ')
		st.write('You must enter something.')
		return False
	else:
		return True

# if the text is devided in sentences, 
# the text will be split and transliterated sentence by sentence 
def split_n_translit(txt):
	lst = txt.split('.')
	blanks = lst.count('')
	if blanks > 0:
		for i in range(blanks):
			lst.remove('')
	text = []
	if lst == []:
		st.write('There is something wrong with your text.')
		return False
	else:
		for sentence in lst:
			text.append(translit(sentence, max_length = 1024)+'.')
		st.write(' '.join(text))
		return True

# demo mode	
def demo():
	sample = pd.read_csv('test180k.csv', sep=',').sample(1)
	
	# tg-fa
	for k, row in sample.iterrows():
		target = row.Pers
		prediction = translit(row.Taj, max_length = 1024)
		st.write('input:', row.Taj)
		st.write('target:', target)
		st.write('prediction:', prediction)
		st.write('lev. distance:', distance(target, prediction))
		st.write('lev. ratio:', ratio(target, prediction))
	
	st.write()
	st.write('--------------------------------------------')
	st.write()	
	
	# fa-tg
	for k, row in sample.iterrows():
		target = row.Taj
		prediction = translit(row.Pers, max_length = 1024)
		st.write('input:', row.Pers)
		st.write('target:', target)
		st.write('prediction:', prediction)
		st.write('lev. distance:', distance(target, prediction))
		st.write('lev. ratio:', ratio(target, prediction))
	
# title
st.title('Tg-Fa transliterator')

# text area
txt = st.text_area('Enter your text:', placeholder = 'Your text must have less than 500 symbols.')

# transliteration sequence
if st.button('Transliterate'):
	if check_length(txt) == True:
		st.write('Output: ')
		if txt.count('.') > 0:
			split_n_translit(txt)
		else:
			st.write(translit(txt, max_length = 1024))	
