#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:12:24 2020

@author: toshitt
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
        "I love my dog",
        "I love my cat",
        "You love my dog!",
        "Ugh!I hate dogs!",
        "Do you like my new dog?"]

tokenizer=Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(sentences)

test_data=["I love cigars",
           "My dog loves chocolates"]
test_sequences=tokenizer.texts_to_sequences(test_data)
padded=pad_sequences(sequences,padding='post',truncating='post',maxlen=6)
padded_test=pad_sequences(test_sequences,padding='post')
#print(word_index)
#print(sequences)
#print(test_sequences)
#print(padded_test)