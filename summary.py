from typing import List

# import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import SnowballStemmer
import re
import numpy as np
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# TODO quoting problem: e.g.
# "This way, tax-i™ makes legal research many times more complete and efficient.’ Roadmap for tax-i™ tax-i™
# currently contains ..... is considered as one sentence in the algorithm


class Summarize:
    def __init__(self, text: str, lang: str = "english"):
        self.text = text
        self.lang = lang

    @staticmethod
    def sent_tokenizer(text: str) -> List:
        """Tokenize into lists of sentences, with primary processes eg ignore
        citations.

        :param text: raw text
        :return: a list of sentences
        """
        # TODO possible solutions for sth like "Rev.", "Ave."...:
        # 1. add punkt_param of abbr. manually
        # https://stackoverflow.com/questions/34805790/how-to-avoid-nltks-sentence-tokenizer-splitting-on-abbreviations
        # 2. use regex sth like "upper letter + many lower letter + ."

        # For acronyms, like i.e., F.B.I., ..., add a "<" sign after such that sent_tokenize will not stop sentences
        acronyms_low = "([a-z][.][a-z][.](?:[a-z][.])?)"
        acronyms_up = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        text = re.sub(acronyms_low, "\\1<", text)
        text = re.sub(acronyms_up, "\\1<", text)

        # use default pretrained model from nltk, know most of the prefixes, suffixes
        # but not Rev. etc., and so forth
        tokens = sent_tokenize(text)
        sentences = []
        for token in tokens:
            token = re.sub("<", " ", token)  # eliminate "<" sign(s) in sentences
            # deal with python input which may included "\n" and +
            token = re.sub("\n", " ", token)
            token = re.sub(" +", " ", token)
            # get rid of ', ", ), ]... at the beginning.
            # TODO '...' he said.
            if token[0] in [
                '"',
                "'",
                ")",
                "]",
            ]:
                token = token[1:]
            # get rid of citations at the beginning and/or ending(1-99)
            # TODO shorter way anything between "[]"
            if token[0] == "[" and token[2] == "]":
                token = token[3:]
            if token[0] == "[" and token[3] == "]":
                token = token[4:]
            if token[-2] == "]" and token[-4] == "[":
                token = token[:-4] + token[-1]
            if token[-2] == "]" and token[-5] == "[":
                token = token[:-5] + token[-1]
            if token[0] == " ":  # extra space at the beginning
                token = token[1:]
            token.strip()
            sentences.append(token)
        return sentences

    def preprocess(self, sentences: List) -> List:
        """Preprocesses after tokenized: 1) removes stopwords 2) removes
        special characters 3) removes and leading and trailing spaces 4)
        transforms all words to lowercase.

        :param sentences: list of sentences after tokenized
        :return: 2D list of words in sentences e.g. [[w,w,w],[w,w,w]]
        """
        tokens = sentences
        # stopwords e.g. i, he, she, they, it, 's, 've, 'd ....
        stop_words = stopwords.words(self.lang)
        preprocessed_sentences = []
        for index, s in enumerate(tokens):
            preprocessed_sentences.append([])
            words = word_tokenize(s)
            for word in words:
                word = re.sub("\n", " ", word)
                # remove special characters, not done in Smmry
                # e.g. fig1, $ @..., but no Covid-19
                # word = re.sub("[^A-Za-z]+", " ", word)
                word = re.sub(" +", " ", word)
                word = word.strip()
                word = word.lower()  # transforms to lowercase
                if word and word not in stop_words:
                    preprocessed_sentences[index].append(word)
        return preprocessed_sentences

    def tag_pos(self, preprocessed_sentences: List, pos=None) -> List:
        """Reference https://www.nltk.org/book/ch05.html Keep relevant words by
        using `tag` which specify parts of speech of words Based on Smmry, we
        want noun(NN), past participle verb(VBN, considered passive voice), and
        adjectives(JJ)

        :param preprocessed_sentences: list of sentences after preprocessed
        :param pos: Parts Of Speech that is needed
        :return: 2D list of tagged words of sentences
        """
        if pos is None:
            # based on Smmry, highlight n.(NN), past participle v.(VBN, considered passive voice), and adj.(JJ)
            pos = ["NN", "VBN", "JJ"]
        tagged_words_sentences = []
        for index, sentence in enumerate(preprocessed_sentences):
            tagged_words_sentences.append([])
            for word in sentence:
                # tag word with its p-o-s, e.g. [('macalester', 'NN')]
                word = pos_tag([word])
                if word[0][1] in pos:
                    tagged_words_sentences[index].append(word[0][0])
        return tagged_words_sentences

    def stem(self, tagged_words_sentences: List) -> List:
        """Stem words.

        :param tagged_words_sentences: 2D list of tagged words of sentences
        :return: 2D list of stemmed and tagged words of sentences
        """
        stemmer = SnowballStemmer(self.lang)
        stemmed_sentences = []
        for index, sentences in enumerate(tagged_words_sentences):
            stemmed_sentences.append([])
            for word in sentences:
                word = stemmer.stem(word)
                stemmed_sentences[index].append(word)
        return stemmed_sentences

    @staticmethod
    def build_similarity_matrix(stemmed_tokens: List) -> np.ndarray:
        """
        Reference: https://monkeylearn.com/blog/what-is-tf-idf/
        Creates tfidf vector using sklearn.feature_extraction.text.TfidfVectorizer() and builds pairwise similarity
        matrix of linear kernal using sklearn.metrics.pairwise.linear_kernel().
        :param stemmed_tokens: list of stemmed words of sentences
        :return: pairwise similarity matrix
        """
        token_strings = [" ".join(sentence) for sentence in stemmed_tokens]
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.50)
        X = vectorizer.fit_transform(token_strings)
        cosine_similarities = linear_kernel(X, X)
        for index1 in range(len(cosine_similarities)):
            for index2 in range(len(cosine_similarities)):
                if index1 == index2:
                    cosine_similarities[index1][index2] = 0
        for index in range(len(cosine_similarities)):
            if cosine_similarities[index].sum() == 0:
                continue
            else:
                cosine_similarities[index] /= cosine_similarities[index].sum()
        return cosine_similarities

    @staticmethod
    def text_rank(
            similarity_matrix: np.ndarray, eps: float = 0.0001, p: float = 0.85
    ) -> np.ndarray:
        """
        reference: https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
        :param similarity_matrix: input similarity_matrix
        :param eps: endurable error for iterations
        :param p: greedy probability i.e. 1-p for randomly randomly choosing
        :return: np.array of unsorted ranked sentences' values
        """
        P = np.ones(len(similarity_matrix)) / len(similarity_matrix)
        while True:
            new_P = np.ones(len(similarity_matrix)) * (1 - p) / len(
                similarity_matrix
            ) + p * similarity_matrix.T.dot(P)
            delta = abs(new_P - P).sum()
            if delta <= eps:
                return new_P
            P = new_P

    def summarize(self, length: int = 5) -> str:
        """
        Pipeline:
        1) sent_tokenizer(),
        2) preprocessing(),
        3) tag_pos(),
        4) stem(),
        5) build_similarity_matrix(),
        6) text_rank()
        :param length: number of sentences that user want to after summarized
        :return: summarized sentences
        """
        original_sentences = self.sent_tokenizer(self.text)
        preprocessed_tokens = self.preprocess(original_sentences)
        tagged_tokens = self.tag_pos(preprocessed_tokens)
        # tag first because after tag  v. -> n.
        stemmed_tokens = self.stem(tagged_tokens)
        sentence_ranks = self.text_rank(self.build_similarity_matrix(stemmed_tokens))
        ranked_sentence_indexes = [
            item[0]
            for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])
        ]
        selected_sentences = sorted(ranked_sentence_indexes[:length])
        # make sure chronological order
        summary = itemgetter(*selected_sentences)(original_sentences)
        # a line of space between each output sentence
        out_summary = "\n\n".join(summary)
        return out_summary
