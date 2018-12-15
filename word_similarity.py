import os
import sys
import math
import nltk
from nltk.corpus import wordnet as wn

class WordSimilarity:

    def __init__(self, alpha=0.2, beta=0.45):
        """
        Initialize variables. Parameters initialized according to the paper.
        """

        self.alpha = alpha
        self.beta = beta

        self.max_path_length = 20.0

        # Change to nltk.stopwords?
        # self.common_words = ['the', 'a', 'in', 'on', 'to', 'is']


    def get_best_synset_pair(self, word1, word2, word1_tag=None, word2_tag=None):
        """
        Returns the best synset pair with the highest similaity among all pairs

        Args:
            word1: source word
            word2: Word compared to
            word1_tag: POS tag of word1
            word2_tag: POS tag of word 2

        Returns:
            Tuple of (synset1, synset2)
        """
        max_similarity = -1.0
        best_pair = None, None
        word1_synsets = wn.synsets(word1, word1_tag)
        word2_synsets = wn.synsets(word2, word2_tag)
        if word1_synsets is None or word2_synsets is None:
            return best_pair
        for syn1 in word1_synsets:
            for syn2 in word2_synsets:
                # Compare pos tags for both words here
                if syn1._pos != 's' and syn1._pos == syn2._pos:
                    sim = wn.path_similarity(syn1, syn2)
                    if sim is None:
                        # print("here sim is None")
                        return None, None
                    elif sim > max_similarity:
                        max_similarity = sim
                        best_pair = syn1, syn2
        return best_pair

    def get_path_similarity(self, syn1, syn2):
        """
        Returns the similarity based on the shortest path length between the
        two words' synsets.

        Args:
            syn1: synset of word1
            syn2: synset of word2

        Returns:
            float, path similarity
        """
        if syn1 is None or syn2 is None:
            return 0.0
        # If they're in  the same synset, return 1
        if syn1 == syn2:
            return 1.0
        # See if there is overlap in the two synsets, dist=1.0
        words1 = set([str(x.name()) for x in syn1.lemmas()])
        words2 = set([str(x.name()) for x in syn2.lemmas()])
        if words1.intersection(words2):
            dist = 1.0
            # return math.exp(-self.alpha*dist)
        # Now we can compute the path length between the two synsets if needed
        else:
            # min_length = sys.maxsize
            dist = syn1.shortest_path_distance(syn2)
            if dist is None:
                return self.max_path_length
            dist += 1
        return math.exp(-self.alpha*dist)


    def get_depth_similarity(self, syn1, syn2):
        """
        Returns similarity based on depth of the model to capture the fact that
        words at upper layers of hierarchical semantic nets have more general
        concepts and less semantic similarity between words than words at lower
        layers.

        Args:
            syn1: synset of word 1
            syn2: synset of word 2

        Returns:
            float, depth similarity
        """

        if syn1 is None or syn2 is None:
            return 0.0
        if syn1 == syn2:
            dist = max([x[1] for x in syn1.hypernym_distances()])
        else:
            # Find the subsumer at the lowest depth and return its depth
            h1 = dict(syn1.hypernym_distances())
            h2 = dict(syn2.hypernym_distances())
            common_subsumers = set(h1).intersection(set(h2))
            if len(common_subsumers) == 0:
                dist = 0.0
            else:
                arr_dist = []
                for node in common_subsumers:
                    dist1 = 0
                    if node in h1:
                        dist1 = h1[node]
                    dist2 = 0
                    if node in h2:
                        dist2 = h2[node]
                    arr_dist.append(max([dist1, dist2]))
                dist = max(arr_dist)

        b1 = math.exp(self.beta*dist)
        b2 = math.exp(-self.beta*dist)
        return (b1 - b2) / (b1 + b2)

    def word_sim(self, word1, word2, word1_tag=None, word2_tag=None):
        """
        Returns word similarity as a product of the path similarity and the
        depth similarity

        Args:
            word1: the primary word
            word2: secondary word to be compared against, could be a word in
                    the synonym list
            word1_tag: pos tag for word 1
            word2_tag: pos tag for word 2

        Returns:
            float, combined word similarity

        """
        (syn1, syn2) = self.get_best_synset_pair(word1, word2, word1_tag, word2_tag)
        return self.get_path_similarity(syn1, syn2) * self.get_depth_similarity(syn1, syn2)

    def get_most_similar_word(self, word, arr):
        """
        Returns the most similar word in arg2 to arg1. Used to find the most
        similar word for the joint word set.

        Args:
            word: word to compare all words in joint set 2
            arr: set of joint words in both sentences

        Returns:
            tuple of word in arg2 that is most similar to arg1 and its combined
            word similarity with that word
        """

        max_similarity = 0.0
        most_sim_word = ""
        for w in arr:
            sim = self.word_sim(word, w)
            if sim > max_similarity:
                max_similarity = sim
                most_sim_word = w
        return (most_sim_word, max_similarity)
