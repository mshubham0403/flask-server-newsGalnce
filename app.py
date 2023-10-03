


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize

import re
from flask import Flask,request,jsonify
import nltk
from flask_cors import CORS


import requests


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




app = Flask(__name__)
CORS(app)

PORT=8087





@app.route('/',methods = ['GET','POST'])
def base():
	return 'summarizer for News Glance'


@app.route('/getSummary', methods=['POST'])
def getSummary():
	try:
		print("requesreached")
		req = request.get_json(force=True)
		print("-"*80)
		content,requiredLength = req.get('content'),req.get('finalSize')
		if not content:
			return jsonify({'error': 'Missing articleText field in request'}), 404
		summary = driver(content,requiredLength)
		return jsonify({'summary': summary}),200

	except Exception as e:
		return jsonify({'error': 'An error occurred: ' + str(e)}), 500
			
	


def clean(sentences):
	lemmatizer = WordNetLemmatizer()
	cleaned_sentences = []
	for sentence in sentences:
		sentence = sentence.lower()
		sentence = re.sub(r'[^a-zA-Z]',' ',sentence)
		sentence = sentence.split()
		sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in set(stopwords.words('english'))]
		sentence = ' '.join(sentence)
		cleaned_sentences.append(sentence)
	return cleaned_sentences

def init_probability(sentences):
	probability_dict = {}
	words = word_tokenize('. '.join(sentences))
	total_words = len(set(words))
	for word in words:
		if word!='.':
			if not probability_dict.get(word):
				probability_dict[word] = 1
			else:
				probability_dict[word] += 1

	for word,count in probability_dict.items():
		probability_dict[word] = count/total_words 
	
	return probability_dict

def update_probability(probability_dict,word):
	if probability_dict.get(word):
		probability_dict[word] = probability_dict[word]**2
	return probability_dict

def average_sentence_weights(sentences,probability_dict):
	sentence_weights = {}
	for index,sentence in enumerate(sentences):
		if len(sentence) != 0:
			average_proba = sum([probability_dict[word] for word in sentence if word in probability_dict.keys()])
			average_proba /= len(sentence)
			sentence_weights[index] = average_proba 
	return sentence_weights

def generate_summary(sentence_weights,probability_dict,cleaned_article,tokenized_article,summary_length = 30):
	summary = ""
	current_length = 0
	while current_length < summary_length :
		highest_probability_word = max(probability_dict,key=probability_dict.get)
		sentences_with_max_word= [index for index,sentence in enumerate(cleaned_article) if highest_probability_word in set(word_tokenize(sentence))]
		sentence_list = sorted([[index,sentence_weights[index]] for index in sentences_with_max_word],key=lambda x:x[1],reverse=True)
		summary += tokenized_article[sentence_list[0][0]] + "\n"
		for word in word_tokenize(cleaned_article[sentence_list[0][0]]):
			probability_dict = update_probability(probability_dict,word)
		current_length+=1
	return summary

def driver(article,required_length):
	required_length = int(required_length)
	tokenized_article = sent_tokenize(article)
	cleaned_article = clean(tokenized_article) 
	probability_dict = init_probability(cleaned_article)
	sentence_weights = average_sentence_weights(cleaned_article,probability_dict)
	summary = generate_summary(sentence_weights,probability_dict,cleaned_article,tokenized_article,required_length)
	return summary


if __name__ == "__main__":
    app.run(port=PORT)
