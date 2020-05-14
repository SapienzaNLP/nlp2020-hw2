import json
import random

import numpy as np
from typing import List, Tuple

from model import Model


def build_model_34(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return Baseline()

def build_model_234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    return Baseline(return_predicates=True)
    # raise NotImplementedError

def build_model_1234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    return Baseline(return_predicates=True)
    # raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, return_predicates=False):
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence['pos_tags']:
            prob = self.baselines['predicate_identification'][pos]['positive'] / self.baselines['predicate_identification'][pos]['total']
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)
        
        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(zip(sentence['lemmas'], predicate_identification)):
            if not is_predicate or lemma not in self.baselines['predicate_disambiguation']:
                predicate_disambiguation.append('_')
            else:
                predicate_disambiguation.append(self.baselines['predicate_disambiguation'][lemma])
                predicate_indices.append(idx)
        
        argument_identification = []
        for dependency_relation in sentence['dependency_relations']:
            prob = self.baselines['argument_identification'][dependency_relation]['positive'] / self.baselines['argument_identification'][dependency_relation]['total']
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(sentence['dependency_relations'], argument_identification):
            if not is_argument:
                argument_classification.append('_')
            else:
                argument_classification.append(self.baselines['argument_classification'][dependency_relation])
        
        if self.return_predicates:
            return {
                'predicates': predicate_disambiguation,
                'roles': {i: argument_classification for i in predicate_indices},
            }
        else:
            return {'roles': {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path='data/baselines.json'):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "pos_tags":
                            ["IN", "DT", "NN", ",", "NNP", "NNP", "CC", "DT", "NNS", "VBP", "IN", "DT", "JJ", "NNS", "IN", "VBG", "DT", "NN", "NN", "VBP", "RB", "VBN", "VBN", "."],
                        "dependency_heads":
                            ["10", "3", "1", "10", "6", "10", "6", "9", "7", "0", "10", "14", "14", "20", "14", "15", "19", "19", "16", "11", "20", "20", "22", "10"],
                        "dependency_relations":
                            ["ADV", "NMOD", "PMOD", "P", "TITLE", "SBJ", "COORD", "NMOD", "CONJ", "ROOT", "OBJ", "NMOD", "NMOD", "SBJ", "NMOD", "PMOD", "NMOD", "NMOD", "OBJ", "SUB", "TMP", "VC", "VC", "P"],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "pos_tags": [...], # SAME AS BEFORE
                        "dependency_heads": [...], # SAME AS BEFORE
                        "dependency_relations": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence. 
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence. 
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence. 
                    }
        """
        pass
