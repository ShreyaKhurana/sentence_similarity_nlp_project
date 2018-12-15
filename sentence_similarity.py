import math
import numpy as np
import nltk
from nltk.corpus import wordnet as wn, brown

import word_similarity


class SentenceSimilarity:

    def __init__(self, word_sim, delta=0.85):
        self.word_order_thresh = 0.4
        self.semantic_thresh = 0.2
        self.delta = delta
        self.word_sim = word_sim
        self.brown_freq = {}
        self.N = 0
        self.computeInfoContent()

    def computeInfoContent(self):
        """
        Computes the information content of each word in the Brown Corpus using
        Laplace add-1 smoothing

        """

        for sen in brown.sents():
            for word in sen:
                self.N += 1
                word = word.lower()
                if word not in self.brown_freq:
                    self.brown_freq[word] = 1
                else:
                    self.brown_freq[word] += 1


    def get_info_content(self, word):
        """
        Returns the information content of the word in the Brown Corpus

        Args:
            word: word to compute info content for

        Returns:
            float, info content of arg1
        """
        word = word.lower()
        if word in self.brown_freq:
            return 1.0 - math.log(self.brown_freq[word] + 1) / math.log(self.N+1)
        else:
            return 1.0


    def get_semantic_vector(self, sentence, joint_word_set):
        """
        Computes the semantic vector of the sentence in arg1

        Args:
            sentence: tokenized list of words
            joint_word_set: set of words in both the sentences

        Returns:
            list of length(arg2)

        """

        vec = [0.0]*len(joint_word_set)
        # print(joint_word_set)
        for i, jw in enumerate(joint_word_set):
            if jw in sentence:
                vec[i] = 1.0 * self.get_info_content(jw) * self.get_info_content(jw)
            else:
                most_similar_word, highest_sim_score = self.word_sim.get_most_similar_word(jw, sentence)
                # print(most_similar_word, highest_sim_score)
                if highest_sim_score > self.semantic_thresh:
                    vec[i] = highest_sim_score * self.get_info_content(jw) * self.get_info_content(most_similar_word)
                else:
                    vec[i] = 0.0

        return vec

    def get_semantic_similarity(self, sentence_1, sentence_2):
        """
        Returns the semantic similarity of the two semantic vectors

        Args:
            sentence_1: first sentence, tokenized, list
            sentence_2: second sentence, tokenized, list

        Returns:
            float, semantic similarity
        """
        joint_word_set = set(sentence_1).union(set(sentence_2))

        sem_vec1 = np.array(self.get_semantic_vector(set(sentence_1), joint_word_set))
        sem_vec2 = np.array(self.get_semantic_vector(set(sentence_2), joint_word_set))
        # print(sem_vec1)
        return np.dot(sem_vec1, sem_vec2.T) / (np.linalg.norm(sem_vec1) * np.linalg.norm(sem_vec2))


    def get_word_order(self, sentence_dict, joint_word_set):
        """
        Computes the word order vector for a sentence to differentiate between
        the sentences having the same word set but different orders.

        Args:
            sentence: dict of tokenized sentence words - word:index
            joint_word_set: dict of word:index in joint word set

        Returns:
            arr of length joint_word_set, word order vector
        """
        word_order_vec = [0]*len(joint_word_set)
        for i, jw in enumerate(joint_word_set):
            if jw in sentence_dict:
                word_order_vec[i] = sentence_dict[jw]
            else:
                most_similar_word, highest_sim_score = self.word_sim.get_most_similar_word(jw, list(sentence_dict.keys()))
                if highest_sim_score > self.word_order_thresh:
                    word_order_vec[i] = sentence_dict[most_similar_word]
        return word_order_vec

    def get_word_order_similarity(self, sentence_1, sentence_2):
        """
        Computes S.r i.e the word order similarity of two sentences

        Args:
            sentence_1: Sentence 1, tokenized, list
            sentence_2: sentence_2, tokenized, list

        Returns:
            float, word order similarity
        """
        # Add 1 because they start indexing from 1 in the paper
        sent_dict_1 = {word:id+1 for id, word in enumerate(sentence_1)}
        sent_dict_2 = {word:id+1 for id, word in enumerate(sentence_2)}
        joint_word_set = set(sentence_1).union(set(sentence_2))
        word_order_1 = np.array(self.get_word_order(sent_dict_1, joint_word_set))
        word_order_2 = np.array(self.get_word_order(sent_dict_2, joint_word_set))

        a = np.linalg.norm(word_order_1 - word_order_2)
        b = np.linalg.norm(word_order_1 + word_order_2)

        return 1.0 - a / b

    def get_sentence_similarity(self, sentence_1, sentence_2):
        """
        Get combined sentence similarity based on word similarity, semantic
        sentence similarity and word order similarity

        Args:
            sentence_1: sentence, string, non-tokenized
            sentence_2: sentence, string, non-tokenized

        Returns:
            float, combined sentence similarity
        """
        sentence_1 = nltk.word_tokenize(sentence_1)
        sentence_2 = nltk.word_tokenize(sentence_2)

        s_r = self.get_word_order_similarity(sentence_1, sentence_2)
        s_s = self.get_semantic_similarity(sentence_1, sentence_2)
        # print("Semantic similarity is ", s_s)
        # print("Word order similarity is ", s_r)
        return self.delta * s_s + (1-self.delta) * s_r
