# Statistical-Grammer-Checker

## Table of Content
  * [Business Problem Statement](#Business-Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Model Building and Tuning](#Model-Building-and-Tuning)
  * [Other used Techniques](#Other-Used-Techniques)
  * [Demo](#demo)
  * [Run project in your local machine](#Run-the-project-in-your-local-machine)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


## Business Problem Statement

## Data
Data Source : P

## Used Libraries and Resources
**Python Version** : 3.6

**Libraries** : sklearn, pandas, numpy, matplotlib, seaborn, flask, json, pickle

**References** : https://towardsdatascience.com/, https://machinelearningmastery.com/


## Data Preprocessing
I Preprocessed this data with the following steps:

1. Split data into sentences using "\n" as the delimiter.
2. Split each sentence into tokens.

Note : The goal is to check for grammer errors. We assume that the training data/corpus is grammatically correct. That's why we should not perform any preprocessing technique to our training data/corpus in order to keep it unchanged. 


## Model Building (Developing a bigram language model)

* A n-gram language model assumes that the probability of the next word depends only on previous n-1 grams. (The previous n-1 gram is the series of the previous 'n-1' words/tokens). In case of bigram language model the probability of the next word depends on the previous gram/word.
* - The conditional probability for the word at position 't' in the sentence, given that the words preceding it is :
* ![alt text](https://github.com/Lori10/Statistical-Grammer-Checker-FromScratch/blob/main/img1.PNG "Image")
* We can estimate this probability  by counting the occurrences of these series of words in the training data. The probability can be estimated as a ratio, where the numerator is the number of times word 't' appears after words t-1 till t-n in the training data/corpus and the denominator is the number of times word t-1 till t-n appears in the training data/corpus. In the bigram model, the numerator would be the number of times word 't' appears after words t-1 in the training data/corpus and the denominator is the number of times word t-1 appears in the training data/corpus. The function C() denotes the number of occurence of the given sequence. 
* We should keep in mind
* This equation tells us that to estimate probabilities based on n-grams, we need the counts of n-grams (for denominator) and (n-1)-grams (for numerator). For example if we want to use trigram we need to calculate bigram and trigram counts. In case of using bigram model, we have to calculate bigram und unigram counts.

* When computing the counts for n-grams,  we prepare the sentence beforehand by prepending n-1 starting markers "<s\>" to indicate the beginning of the sentence and 1 ending marker "<e\>" to indicate the end of the sentence. This modification is done because we also want to calculate the probability of beginnig and ending the sentence given a particular word. Only when calculating bigram counts, we should prepend 2 "<s\>" markers and not 1 in order to include counting the occurance of the pair ("<s\>", "<s\>"). This is done because in case of using a trigram language model we would have in the beginnig of each sentence 2 markers "<s\>" "<s\>". When calculating the conditional probabilities for the word that are placed in the beginning of the sentence we need to know the occurance of ("<s\>", "<s\>") which is not found in the bigram counts since there we used only 1 "<s\>" marker. So, for example in case of trigram analysis if the tokenized sentence is ["I" , "like", "food"], modify it to be ["<s\>" , "<s\>", "I", "like", "food"]. We also must prepare the sentence for counting by appending an end token "<e\>" so that the model can predict when to finish a sentence. 
- We will store the counts as a dictionary. The key of each key-value pair in the dictionary is a **tuple** of n words (and not a list). The value in the key-value pair is the number of occurrences.  - The reason for using a tuple as a key instead of a list is because a list in Python is a mutable object (it can be changed after it is first created).  A tuple is "immutable", so it cannot be altered after it is first created.  This makes a tuple suitable as a data type for the key in a dictionary.

| Model Name        | Deafult Model Test Score |Default Model Training Score | Default Model CV Score | Tuned Model Test Score | Tuned Model Training Score | Tuned Model CV Score | 
|:-----------------:|:------------------------:|:---------------------------:|:----------------------:|:----------------------:|:--------------------------:|:---------------------:|
|Linear Regression  |     0.7891               |     0.7833                  |         0.7800         |      0.7891            |           0.7833           |     0.7800             |
|Random Forest      |     0.8794               |     0.9700                  |         0.8758         |      0.8793            |           0.7833           |     0.8792            |
|KNN                |     0.8514               |     0.8861                  |         0.8105         |      0.8504            |           0.9824           |  0.8248              |


## Other Used Techniques

* Object oriented programming is used to build this project in order to create modular and flexible code.
* Built a client facing API (web application) using Flask.
* A retraining approach is implemented using Flask framework.
* Using Logging every information about data cleaning und model training HISTORY (since we may train the model many times using retraining approach)  is stored is some txt files and csv files for example : the amount of missing values for each feature, the amount of records removed after dropping the missing values and outliers, the amount of at least frequent categories labeled with 'other' during encoding, the dropped constant features, highly correlated independent features, which features are dropping during handling multicolleniarity, best selected features, model accuracies and errors etc.

## Demo

This is how the web application looks like : 


![alt text](https://github.com/Lori10/Banglore-House-Price-Prediction/blob/master/Project%20Code%20Pycharm/demo_image.jpg "Image")



## Run the project in your local machine 

1. Clone the repository
2. Open the project directory (PyCharm Project Code folder) in PyCharm  and create a Python Interpreter using Conda Environment : Settings - Project : Project Code Pycharm - Python Interpreter - Add - Conda Environment - Select Python Version 3.6 - 
3. Run the following command in the terminal to install the required packages and libraries : pip install -r requirements.txt
4. Run the file app.py by clicking Run and open the API that shows up in the bottom of terminal.


## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Lori10/Banglore-House-Price-Prediction/issues) here by including your search query and the expected result

## Future Scope
* Optimize Flask app.py Front End
