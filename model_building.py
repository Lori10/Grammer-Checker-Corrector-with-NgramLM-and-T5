from preprocessing import Preprocessor
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix


class ModelBuilding:
    def __init__(self):
        pass

    def estimate_conditional_probability(self, word, previous_n_minus1_gram,
                                         n_minus1_gram_counts, n_gram_counts, vocabulary_size, k=1):
        """
        Estimate the probabilities of a next word using the ngram counts with k-smoothing

        Args:
            word: next word (string)
            previous_n_minus1_gram: A sequence of words of length n-1 (list)
            n_minus1_gram_counts: Dictionary of counts of (n-1)grams
            n_gram_counts: Dictionary of counts of n-grams
            vocabulary_size: number of words in the vocabulary (integer or double)
            k: positive constant, smoothing parameter (integer or double)

        Returns:
            A probability
        """
        # convert list to tuple to use it as a dictionary key
        previous_n_minus1_gram = tuple(previous_n_minus1_gram)

        # Set the denominator
        # If the previous n-gram exists in the dictionary of n-gram counts,
        # Get its count.  Otherwise set the count to zero
        # Use the dictionary that has counts for n-grams
        if previous_n_minus1_gram in n_minus1_gram_counts:
            previous_n_minus1_gram_count = n_minus1_gram_counts[previous_n_minus1_gram]
        else:
            previous_n_minus1_gram_count = 0

        # Calculate the denominator using the count of the previous n gram
        # and apply k-smoothing

        denominator = previous_n_minus1_gram_count + k * vocabulary_size

        # Define n gram as the previous n-gram plus the current word as a tuple
        n_gram = previous_n_minus1_gram + (word,)

        # Set the count to the count in the dictionary,
        # otherwise 0 if not in the dictionary
        # use the dictionary that has counts for the n-gram plus current word
        n_gram_count = n_gram_counts[n_gram] if n_gram in n_gram_counts else 0

        # Define the numerator use the count of the n-gram plus current word,
        # and apply smoothing

        numerator = n_gram_count + k

        probability = numerator / denominator

        return probability

    def ngram_language_model(self, n, test_sents, dic_n_grams, threshold, nr_unique_words, k=1):
        """
        Estimate the probability of sentences using trigram language model

        Args:
            sents: raw sentences that need to be tested for grammar errors (list)
            bigram_counts: bigram_counts that were calculated in the training data/corpus (dictionary)
            trigram_counts: trigram_counts that were calculated in the training data/corpus (dictionary)
            threshold: threshold which determines when we classify a sentence as correct or incorrect (integer or double)
            vocabulary_size: number of words in the vocabulary (integer or double)
            k: positive constant, smoothing parameter (integer or double)

        Returns:
            output_dic : A dictionary which contains as tuple the raw sentence and as value a list of probability of
            that sentence, string which shows if the sentence is grammatically correct or not
            binary_predictions : a list which contains 0s and 1s for each sentence. 0 means the sentence was grammatically correct and 1 incorrect

        """

        preprocessor = Preprocessor()
        tokenized_test_sentences = preprocessor.tokenize_sentences(test_sents)

        nminus1_gram_counts = dic_n_grams[n - 1]
        n_gram_counts = dic_n_grams[n]

        output_dic = {}
        binary_predictions = []

        for (sen, raw_sen) in zip(tokenized_test_sentences, test_sents):
            sentence = ['<s>'] * n + sen + ['<e>']
            sentence_prob = 1

            for i in range(n - 1, len(sentence)):
                list_previous_tokens = []
                for j in range(n - 1, 0, -1):
                    list_previous_tokens.append(sentence[i - j])

                cond_prob = self.estimate_conditional_probability(sentence[i], list_previous_tokens, nminus1_gram_counts,
                                                             n_gram_counts, nr_unique_words, k)
                sentence_prob = sentence_prob * cond_prob

            if sentence_prob >= threshold:
                output_dic[raw_sen] = ['Sentence grammatically correct', sentence_prob]
                binary_predictions.append(1)
            else:
                output_dic[raw_sen] = ['Sentence grammatically incorrect', sentence_prob]
                binary_predictions.append(0)

        return output_dic, binary_predictions


    def hyperparameter_tuning(self, n_value, threshold_values, k_values, dic_ngram_counts, test_sentences, true_labels,
                              nr_unique_words):
        scores = {}
        for thres in threshold_values:
            for k in k_values:
                output_dic, predictions = self.ngram_language_model(n_value, test_sentences, dic_ngram_counts, thres, nr_unique_words, k)

                # get auc score
                auc_score = roc_auc_score(true_labels, predictions)

                # get F1 score of class 0 (grammatically incorrect sentences)
                f1_score_class0 = f1_score(true_labels, predictions, pos_label=0)

                # get recall of class 0 (grammtically incorrect sentences)
                recall_class0 = recall_score(true_labels, predictions, pos_label=0)

                # get confusion matrix / false positive
                tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

                # store the threhold and its auc score and false positive amount
                scores[(thres, k)] = [auc_score, f1_score_class0, recall_class0, fp]

        # decide which metrics to use for choosing the best model (this case auc score)
        max_score = max([el[1] for el in scores.values()])
        tuned_hyperparameters_dic = [(hyperparameters, list_scores) for (hyperparameters, list_scores) in scores.items()
                                     if list_scores[1] == max_score]

        return scores, tuned_hyperparameters_dic