"""
Hidden Markov Model (HMM) Part-of-Speech (POS) Tagger with Viterbi Decoding.
"""

from collections import Counter, defaultdict
import math
from typing import Dict, List, Optional, Set, Tuple


class HiddenMarkovModelTagger:
    """
    HMM POS Tagger with Viterbi dynamic programming decoding.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.tags: Set[str] = set()
        self.words: Set[str] = set()
        self.initial_counts: Counter = Counter()
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self.emission_counts: Dict[str, Counter] = defaultdict(Counter)
        self.tag_counts: Counter = Counter()

    def fit(self, tagged_sentences: List[List[Tuple[str, str]]]) -> "HiddenMarkovModelTagger":
        """
        Fit transition and emission probability parameters from tagged sentences:
        tagged_sentences is a list of sentences, where each sentence is a list of (word, tag) tuples.
        """
        for sentence in tagged_sentences:
            if not sentence:
                continue
            first_word, first_tag = sentence[0]
            self.initial_counts[first_tag] += 1

            for i, (word, tag) in enumerate(sentence):
                w_lower = word.lower()
                self.tags.add(tag)
                self.words.add(w_lower)
                self.tag_counts[tag] += 1
                self.emission_counts[tag][w_lower] += 1

                if i > 0:
                    prev_tag = sentence[i - 1][1]
                    self.transition_counts[prev_tag][tag] += 1

        return self

    def _get_transition_prob(self, prev_tag: str, curr_tag: str) -> float:
        num = self.transition_counts[prev_tag][curr_tag] + self.epsilon
        denom = sum(self.transition_counts[prev_tag].values()) + self.epsilon * len(self.tags)
        return num / denom if denom > 0 else self.epsilon

    def _get_emission_prob(self, tag: str, word: str) -> float:
        w_lower = word.lower()
        num = self.emission_counts[tag][w_lower] + self.epsilon
        denom = self.tag_counts[tag] + self.epsilon * (len(self.words) + 1)
        return num / denom if denom > 0 else self.epsilon

    def viterbi(self, sentence_tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Decode the most probable POS tag sequence using the Viterbi algorithm.
        """
        if not sentence_tokens or not self.tags:
            return [(w, "NOUN") for w in sentence_tokens]

        T = len(sentence_tokens)
        tags_list = sorted(list(self.tags))
        num_tags = len(tags_list)

        viterbi_mat = [{} for _ in range(T)]
        backpointers = [{} for _ in range(T)]

        total_init = sum(self.initial_counts.values()) + self.epsilon * num_tags
        # Step 1: Initialization
        first_word = sentence_tokens[0]
        for tag in tags_list:
            init_prob = (self.initial_counts[tag] + self.epsilon) / total_init
            emiss_prob = self._get_emission_prob(tag, first_word)
            viterbi_mat[0][tag] = math.log(max(init_prob * emiss_prob, 1e-12))
            backpointers[0][tag] = None

        # Step 2: Recursion
        for t in range(1, T):
            word = sentence_tokens[t]
            for curr_tag in tags_list:
                emiss_prob = self._get_emission_prob(curr_tag, word)
                log_emiss = math.log(max(emiss_prob, 1e-12))

                best_log_prob = -float("inf")
                best_prev_tag = tags_list[0]

                for prev_tag in tags_list:
                    trans_prob = self._get_transition_prob(prev_tag, curr_tag)
                    log_trans = math.log(max(trans_prob, 1e-12))
                    score = viterbi_mat[t - 1][prev_tag] + log_trans + log_emiss

                    if score > best_log_prob:
                        best_log_prob = score
                        best_prev_tag = prev_tag

                viterbi_mat[t][curr_tag] = best_log_prob
                backpointers[t][curr_tag] = best_prev_tag

        # Step 3: Termination
        best_last_tag = max(tags_list, key=lambda t_tag: viterbi_mat[T - 1][t_tag])

        # Step 4: Backtracking
        best_path = [best_last_tag]
        for t in range(T - 1, 0, -1):
            best_prev = backpointers[t][best_path[-1]]
            best_path.append(best_prev)

        best_path.reverse()
        return list(zip(sentence_tokens, best_path))
