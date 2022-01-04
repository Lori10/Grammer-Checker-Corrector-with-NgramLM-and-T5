from flask import Flask, request, render_template
from flask_cors import cross_origin
from flask import Response
from functions import count_n_grams, load_data
from preprocessing import Preprocessor
from model_building import ModelBuilding
import json
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['max_ngram'] is not None:
            max = request.json['max_ngram']

        path_dataset = 'en_US.twitter.txt'
        data = load_data(path_dataset)
        preprocessor = Preprocessor()
        tokenized_training_sentences = preprocessor.get_tokenized_data(data)
        nr_unique_words = len(set([y for x in tokenized_training_sentences for y in x]))

        dic_ngram_counts = {}

        for i in range(1, max+1):
            i_gram_counts = count_n_grams(tokenized_training_sentences, i)
            dic_ngram_counts[i] = i_gram_counts

        model_training_infos = {'nr_unique_words': nr_unique_words,
                                'dic_ngram_counts': dic_ngram_counts}
        with open("model_training_infos.pkl", "wb") as file:
            pickle.dump(model_training_infos, file)

        with open("max_ngram.pkl", "wb") as file:
            pickle.dump(max, file)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("N_Grams Counts Built Successfully / Training successfull!!")


@app.route("/tuning", methods=['POST'])
@cross_origin()
def tuneClient():
    try:
            threshold_values = [pow(10, i) for i in range(-20, -12, 1)]
            k_values = np.linspace(0.1, 0.9, 8)
            with open('model_training_infos.pkl', 'rb') as file:
                encoded_training_infos = pickle.load(file)
            nr_unique_words = encoded_training_infos['nr_unique_words']
            dic_ngram_counts = encoded_training_infos['dic_ngram_counts']

            test_sentences_labels = [['How are you', 1],
                                     ['I am going home.', 1],
                                     ['I are going here.', 0],
                                     ['This are perfect', 0],
                                     ['This is perfect', 1],
                                     ['I am doing it.', 1],
                                     ['These man is very dangerous', 0],
                                     ['Today i am very tired!', 1],
                                     ['He has been doing this job for many years as a data scientist in finance domain.',
                                      1],
                                     ["You has helped me !", 0],
                                     ["My university are over there", 0],
                                     ["My mom cook pancakes for breakfast.", 0]]

            test_sentences = [sent[0] for sent in test_sentences_labels]
            true_labels = [x[1] for x in test_sentences_labels]

            model_builder = ModelBuilding()
            dic, tuned_hyperparameters_dic = model_builder.hyperparameter_tuning(2, threshold_values, k_values, dic_ngram_counts, test_sentences,
                                                true_labels, nr_unique_words)

            models_df = pd.DataFrame()
            models_df['(Threshold, K)'] = list(dic.keys())
            models_df['AUC_Score'] = [scores[0] for scores in list(dic.values())]
            models_df['F1_Score'] = [scores[1] for scores in list(dic.values())]
            models_df['Recall'] = [scores[2] for scores in list(dic.values())]
            models_df.to_csv('tuning.csv', index=False)
            tuned_threshold = models_df[models_df['AUC_Score'] == max(models_df['AUC_Score'])].tail(1)['(Threshold, K)'].iloc[0][0]
            tuned_k = models_df[models_df['AUC_Score'] == max(models_df['AUC_Score'])].tail(1)['(Threshold, K)'].iloc[0][1]

            return Response(f"Hyperparameter Tuning Successfull!!  Tuned Threshold : {tuned_threshold} , T"
                            f"Tuned K-Smoothing Parameter : {tuned_k}")

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Hyperparameter Tuning Successfull!!")



@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        test_sentence = request.form['input_sentence']
        ngram = int(request.form['ngram'])

        with open('max_ngram.pkl', 'rb') as file:
            max_ngram = pickle.load(file)
        if (ngram >= 2) and (ngram <= max_ngram):
            test_sentences = [test_sentence]

            # Load the model
            # A) load the dictionary which includes the threshold and k-smoothing parameter
            with open('hyperparameters.json', 'r') as myfile:
                encoded_hyperparams_str = myfile.read()
                encoded_hyperparams = json.loads(encoded_hyperparams_str)
            threshold = encoded_hyperparams['threshold']
            k_smoothing_parameter = encoded_hyperparams['k']

            # B) load the nr of unique words (vocabulary size) and the n-grams counts dictionary
            with open('model_training_infos.pkl', 'rb') as file:
                encoded_training_infos = pickle.load(file)
            nr_unique_words = encoded_training_infos['nr_unique_words']
            dic_ngram_counts = encoded_training_infos['dic_ngram_counts']

            model_builder = ModelBuilding()
            output_dic, predictions = model_builder.ngram_language_model(ngram, test_sentences, dic_ngram_counts,
                                               threshold, nr_unique_words, k_smoothing_parameter)


            return render_template('home.html', prediction_text=output_dic[test_sentence][0])

        else:
            return render_template('home.html', prediction_text='Value of N must be greater than 1 and lower than ' + str(max_ngram+1))

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
