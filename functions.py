
def load_data(data_path):
    with open(data_path, "r", encoding='utf8') as f:
        data = f.read()
    return data


def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words/tokens
        n: number of words in a sequence

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """

    # Initialize dictionary of n-grams and their counts
    n_grams = {}

    if n > 1:
        start_tokens = [start_token] * n
    elif n == 1:
        start_tokens = [start_token]
    else:
        print('N must be greater than 0')

    # Go through each sentence in the data
    for sentence in data:

        # prepend start token n times, and  append <e> one time
        sentence = start_tokens + sentence + [end_token]

        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)

        # Use 'i' to indicate the start of the n-gram
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence.

        for i in range(len(sentence) - n + 1):

            # Get the n-gram from i to i+n
            n_gram = sentence[i:i + n]

            # check if the n-gram is in the dictionary
            if n_gram in n_grams:  # complete this line

                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1

    return n_grams

