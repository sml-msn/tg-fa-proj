import main as m
import pytest
import pandas as pd
from Levenshtein import distance, ratio

# check accuracy tg-fa
def test_translit_acc_tg_fa():
	sample = pd.read_csv('test180k.csv', sep=',').sample(1)
	for k, row in sample.iterrows():
		target = row.Pers
		prediction = m.translit(row.Taj, max_length = 1024)
	assert ratio(target, prediction) > 0.8
	
# check accuracy fa-tg
def test_translit_acc_fa_tg():
	sample = pd.read_csv('test180k.csv', sep=',').sample(1)
	for k, row in sample.iterrows():
		target = row.Taj
		prediction = m.translit(row.Pers, max_length = 1024)
	assert ratio(target, prediction) > 0.8
	
# cheking if the text is too long
def test_check_length_too_long():
	txt = '''
		Бунёдгузори адабиёти форсу тоҷик Абӯабдуллоҳ Рӯдакӣ соли 858 дар деҳаи Панҷрӯд аз тавобеъи Самарқанд (бинобар ин дар баъзе сарчашмаҳо Рӯдакии Самарқандӣ низ меноманд), 
		ки акнун рустое аз ноҳияи Панҷакент дар вилояти Суғд аст, ба дунё омадааст. 
		Таърихи ҳазору сад солаи адабиёти тоҷик бо номи бунёдгузори он устод Рӯдакӣ сахт вобаста аст. 
		Рӯдакиро муосиронаш ва суханварони баъдина бо унвонҳои ифтихорӣ: 
		Одамушуаро форсӣ: آدم الشعرا‎ Қофиласорои назми форсӣ, Соҳибқирони шоирон, Султони шоирон, Мақаддумушуаро ва ҳамсони инҳо ёд мекунанд. 
		Асосгузор ва сардафтари адабиёт аслан маънои онро надорад ки пеш аз дигарон асар эҷод карда бошад.
		'''
	assert m.check_length(txt) == False

# cheking if the text is empty
def test_check_length_blank():
	txt = ''
	assert m.check_length(txt) == False

# checking if the text is of acceptable length
def test_check_length_ok():
	txt = '''
		Бунёдгузори адабиёти форсу тоҷик Абӯабдуллоҳ Рӯдакӣ соли 858 дар деҳаи Панҷрӯд аз тавобеъи Самарқанд (бинобар ин дар баъзе сарчашмаҳо Рӯдакии Самарқандӣ низ меноманд), 
		ки акнун рустое аз ноҳияи Панҷакент дар вилояти Суғд аст, ба дунё омадааст. 
		'''
	assert m.check_length(txt) == True

# checking if the text consists only of dots
def test_split_n_translit_dots_only():
	txt = '......'
	assert m.split_n_translit(txt) == False

# checking if the text consists of separate sentences
def test_split_n_translit_ok():
	txt = '''
		Таърихи ҳазору сад солаи адабиёти тоҷик бо номи бунёдгузори он устод Рӯдакӣ сахт вобаста аст. 
		Рӯдакиро муосиронаш ва суханварони баъдина бо унвонҳои ифтихорӣ: 
		Одамушуаро форсӣ: آدم الشعرا‎ Қофиласорои назми форсӣ, Соҳибқирони шоирон, Султони шоирон, Мақаддумушуаро ва ҳамсони инҳо ёд мекунанд. 
		Асосгузор ва сардафтари адабиёт аслан маънои онро надорад ки пеш аз дигарон асар эҷод карда бошад.
		'''
	assert m.split_n_translit(txt) == True
