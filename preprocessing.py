import nltk

class Preprocessor:
    def __init__(self):
        pass

    def split_to_sentences(self, data):
        """
        Split data by linebreak "\n"

        Args:
            data: str

        Returns:
            A list of sentences
        """

        sentences = data.split('\n')
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if len(s) > 0]

        return sentences

    def tokenize_sentences(self, sentences):
        """
        Tokenize sentences into tokens (words)

        Args:
            sentences: List of strings

        Returns:
            List of lists of tokens
        """

        # Initialize the list of lists of tokenized sentences
        tokenized_sentences = []

        # Go through each sentence
        for sentence in sentences:
            # Convert to lowercase letters
            # sentence = sentence.lower()

            # Convert into a list of words
            tokenized = nltk.word_tokenize(sentence)

            # append the list of words to the list of lists
            tokenized_sentences.append(tokenized)

        return tokenized_sentences

    def get_tokenized_data(self, data):
        """
        Make a list of tokenized sentences

        Args:
            data: String

        Returns:
            List of lists of tokens
        """

        # Get the sentences by splitting up the data
        sentences = self.split_to_sentences(data)

        # Get the list of lists of tokens by tokenizing the sentences
        tokenized_sentences = self.tokenize_sentences(sentences)
        return tokenized_sentences