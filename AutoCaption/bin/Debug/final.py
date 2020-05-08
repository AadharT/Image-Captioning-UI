#DO NOT RUN THIS CELL < ALREADY SAVES
#from os import listdir
#from pickle import dump
#from keras.applications.vgg16 import VGG16
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
#from keras.models import Model
 
# extract features from each photo in the directory
#def extract_features(directory):
	# load the model
#	model = VGG16()
	# re-structure the model
#	model.layers.pop()
#	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
#	print(model.summary())
	# extract features from each photo
#	features = dict()
#	for name in listdir(directory):
		# load an image from file
#		filename = directory + '/' + name
#		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
#		image = img_to_array(image)
		# reshape data for the model
#		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
#		image = preprocess_input(image)
		# get features
#		feature = model.predict(image, verbose=0)
		# get image id
#		image_id = name.split('.')[0]
		# store feature
#		features[image_id] = feature
#		print('>%s' % name)
#	return features
 
# extract features from all images
#directory = '/content/drive/My Drive/dataset/Flickr Images'
#features = extract_features(directory)

#print('Extracted Features: %d' % len(features))
# save to file
#dump(features, open('/content/drive/My Drive/features.pkl', 'wb'))

import string
import sys
argumentList = sys.argv 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping
 
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)
 
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
 
# save descriptions to file, one per line
#def save_descriptions(descriptions, filename):
#	lines = list()
#	for key, desc_list in descriptions.items():
#		for desc in desc_list:
#			lines.append(key + ' ' + desc)
#	data = '\n'.join(lines)
#	file = open(filename, 'w')
#	file.write(data)
#	file.close()
 
#filename = 'C:/Users/Parth Gupta/Desktop/project/Flickr8k_text/Flickr8k.token.txt'
# load descriptions
#doc = load_doc(filename)
# parse descriptions
#descriptions = load_descriptions(doc)
#print('Loaded: %d ' % len(descriptions))
# clean descriptions
#clean_descriptions(descriptions)
# summarize vocabulary
#vocabulary = to_vocabulary(descriptions)
#print('Vocabulary Size: %d' % len(vocabulary))
# save to file
#save_descriptions(descriptions, 'descriptions.txt')



from pickle import load

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load training dataset (6K)
filename = './Flickr_8k.trainImages.txt'
train = load_set(filename)
#print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
#print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('./features.pkl', train)
#print('Photos: train=%d' % len(train_features))

from keras.preprocessing.text import Tokenizer
# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
#print('Vocabulary Size: %d' % vocab_size)





from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from pickle import dump

def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load training dataset (6K)
filename = './Flickr_8k.trainImages.txt'
train = load_set(filename)
#print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
#print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text


# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('./model_19.h5')
# load and prepare the photograph
photo = extract_features(argumentList[1])
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
description = description[8:-6]
print(description)

