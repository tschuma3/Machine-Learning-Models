import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
The Goal of the neural network is to determine whether a movie review is good or bad
"""


#Load data 
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) #num_words=10000 is only getting the 10000 most frequent words

#Creating a mapping
    #Typically will have to do own mapping
word_index = data.get_word_index()

#Gives tuples that have the string and the words in them 
    #The 3 stands for the amount of special characters
word_index = {k:(v + 3) for k, v in word_index.items()}
#Allows to assign unique values
word_index["<PAD>"] = 0 #PAD is to add length to the end to make all the values the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#This will swap all the values and the keys
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#Redefine training and testing data and trim down/normalize
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

#Decoding the training and testing data
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text]) #The "?" is a defult value so the program does not crash

#Only done when no model is saved
"""
#Defining the model
model = keras.Sequential()
    #Embedding 10000 == the amount of words -- 16 == the dimensions
model.add(keras.layers.Embedding(88000, 16)) #Group words that are similar ---- math version explination: finds word vectors for each word that is passed in and passes those vectors to the next layer
model.add(keras.layers.GlobalAveragePooling1D()) #Takes whatever dimension the data is in and puts it to a lower dimension
model.add(keras.layers.Dense(16, activation="relu")) #Squishes everything between 0 and 1
model.add(keras.layers.Dense(1, activation="sigmoid")) #Same as the layer above

model.summary()

    #binary_crossentropy == calculates the difference from example: 0.2 is from 0
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Splitting the data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

    #batch_size == the amount of data fed into the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

#Saving the model
model.save("model.h5")
"""

#Loads the mode
model = keras.models.load_model("model.h5")

#Use this to open a specific text file named "test.txt" that has a review
"""
def review_encode(s):
    encoded = [0]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])
"""

"""
#Predicting
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(f"Prediction: {str(predict[0])}")
print(f"Actual: {str(test_labels[0])}")
print(results)
"""