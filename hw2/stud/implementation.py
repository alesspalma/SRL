import json
import random
import torch
import stanza
import os
import transformers_embedder as tre

import numpy as np
from typing import List, Tuple, Dict

from model import Model
from stud.mymodel import ModHParams, SRLModel
from stud.myvocab import Vocab


def build_model_34(language: str, device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(language, device)


def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=False):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str, device: str):
        # load the specific model for the input language

        # used in case the test dataset doesn't contain pos tags (not clear from predict's method description)
        self.upos2i = {
            "ADJ": 1,
            "ADP": 2,
            "ADV": 3,
            "AUX": 4,
            "CCONJ": 5,
            "DET": 6,
            "INTJ": 7,
            "NOUN": 8,
            "NUM": 9,
            "PART": 10,
            "PRON": 11,
            "PROPN": 12,
            "PUNCT": 13,
            "SCONJ": 14,
            "SYM": 15,
            "VERB": 16,
            "X": 17,
        }  # leave 0-th index for padding

        self.language = language
        self.device = torch.device(device)

        model_folder = "model/"
        words_vocab = Vocab.load(model_folder + "words_vocab" + language + ".json")
        lemmas_vocab = Vocab.load(model_folder + "lemmas_vocab" + language + ".json")
        # predicates and roles vocabularies are in common for all models, due to transfer learning
        self.preds_vocab = Vocab.load(model_folder + "preds_vocabEN.json")
        self.roles_vocab = Vocab.load(model_folder + "roles_vocab.json")

        # choose the language model based on the language we are using
        transformer_model = "roberta-base"
        if language == "FR":
            transformer_model = "camembert-base"
        if language == "ES":
            transformer_model = "PlanTL-GOB-ES/roberta-base-bne"

        self.hparams = ModHParams(
            words_vocab,
            self.preds_vocab,
            lemmas_vocab,
            self.roles_vocab,
            use_transformer=transformer_model,
        )

        # I will instantiate those in the first run of predict method, to allow the server to go up in just 10 seconds
        self.pos_tagger = None
        self.tokenizer = None
        self.model = None

        # to avoid warnings from the tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def unroll_sentence(self, sentence: Dict) -> List[Dict]:
        """Unrolls a sentence into one sample for each predicate, ready
        to be given as input to the model
        Args:
            sentence (Dict): a dictionary that represents an input sentence

        Returns:
            List[Dict]: a list of dictionaries, each representing an input sample with one predicate
        """
        samples = []
        zero = [0]  # used to add padding for [CLS] and [SEP] tokens

        for idx, pred in enumerate(sentence["predicates"]):
            if pred != "_":  # if it's a predicate, prepare the input sample
                new_sample = {"pred_id": idx}

                new_sample["words"] = sentence["words"]

                new_predicate = ["_"] * len(sentence["predicates"])
                new_predicate[idx] = pred
                new_sample["preds"] = torch.unsqueeze(
                    torch.LongTensor(
                        zero
                        + [self.preds_vocab[token] for token in new_predicate]
                        + zero
                    ),
                    0,
                )

                if "pos_tags" in sentence:
                    new_sample["pos"] = torch.unsqueeze(
                        torch.LongTensor(
                            zero
                            + [self.upos2i[token] for token in sentence["pos_tags"]]
                            + zero
                        ),
                        0,
                    )
                else:
                    doc = self.pos_tagger([sentence["words"]])
                    tags = [word.upos for word in doc.sentences[0].words]
                    new_sample["pos"] = torch.unsqueeze(
                        torch.LongTensor(
                            zero + [self.upos2i[token] for token in tags] + zero
                        ),
                        0,
                    )

                samples.append(new_sample)  # append it as a single sample

        if not samples:  # if the sentence doesn't have predicates
            sentence["pred_id"] = -1
            samples.append(sentence)

        return samples

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
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
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
        if self.model is None:
            # initialize pos tagger, tokenizer and model
            stanza.download(
                lang=self.language.lower(), processors="tokenize,pos", verbose=False
            )
            self.pos_tagger = stanza.Pipeline(
                lang=self.language.lower(),
                processors="tokenize,pos",
                tokenize_pretokenized=True,
                verbose=False,
            )

            self.tokenizer = tre.Tokenizer(
                self.hparams.use_transformer, add_prefix_space=True
            )

            self.model = SRLModel(self.hparams).to(self.device)
            self.model.load_state_dict(
                torch.load(
                    "model/best_weights" + self.language + ".pt",
                    map_location=self.device,
                )
            )
            self.model.eval()

        output = {"roles": dict()}

        samples = self.unroll_sentence(sentence)
        for sample in samples:
            if sample["pred_id"] == -1:  # if no predicates, return empty roles
                return output

            # tokenize and move everything to device
            sample["words"] = self.tokenizer(
                sample["words"], return_tensors=True, is_split_into_words=True
            ).to(self.device)
            sample["preds"], sample["pos"] = sample["preds"].to(self.device), sample[
                "pos"
            ].to(self.device)

            logits = self.model(sample)

            # take predictions, flatten, remove [CLS] and [SEP]
            predictions = torch.argmax(logits, -1).view(-1)[1:-1].tolist()
            predictions = [
                self.roles_vocab.i2w[idx] for idx in predictions
            ]  # convert from indices to labels

            # now we fill the output dict
            output["roles"][sample["pred_id"]] = predictions

        return output
