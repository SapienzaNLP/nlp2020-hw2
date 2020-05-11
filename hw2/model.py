from typing import List


class Model:

    def predict(self, sentence):
        """
        A simple wrapper for your model

        Args:
            sentence: a dictionary that represents an input sentence, for example:
            {
                "words": [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
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
                "roles": [
                    [ "Connective", "_", "_", "_", "_", "Agent", "_", "_", "_", "_", "Theme", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_" ],
                    [ "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "Patient", "_", "_", "_", "_", "_" ],
                    [ "_", "_", "_", "_", "_",  "_",  "_",  "_",  "_",  "_",  "_",  "_",  "_",  "Product",  "_",  "_",  "_",  "_",  "_",  "_",  "Time",  "_",  "_",  "_"  ]
                ],
            },

        Returns:
            sentence: the very same input dictionary except for some fields.
                - If you are just doing argument identification and classification, replace the "roles" field with your predictions.
                - If you are also doing predicate identification and disambiguation, replace the "predicates" field with your predictions.
        """
        raise NotImplementedError
