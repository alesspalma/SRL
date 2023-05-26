# let's start with all the imports
# NOTE: part of this code is taken from notebook #6 - POS tagging
import torch
import stanza
import json
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
from myvocab import Vocab


class SRLDataset(Dataset):
    """My Dataset class for the SRL task"""

    def __init__(
        self,
        data_path: str = None,
        words_vocab: Vocab = None,
        use_pos: bool = False,
        use_transformers: bool = False,
    ):
        """constructor of this class
        Args:
            data_path (str, optional): path where to load the whole Dataset, if passed it will have priority. Defaults to None.
            sentences (List[List[str]], optional): if Dataset is already loaded assume is a test set, pass sentences here. Defaults to None.
            words_vocab (Vocab, optional): if Dataset is already loaded assume is a test set,
            so you already have a vocab to pass here, in order to index the test set. Defaults to None.
            use_pos (bool, optional): whether to generate the pos tags or not. Defaults to False.
        """
        # since I'm not interested in going back from index to pos tag, and the upos tagset is fixed, use a plain dictionary and not a Vocab object
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

        self.use_transformers = use_transformers
        # just a flag expressing if dataset has been indexed or not
        self.encoded_samples = False
        self.data_samples = None  # list of dictionaries containing each sample of the original dataset, but here each sample has only one predicate
        self.ground_truths = None  # ground truths as returned by original parse_data

        if data_path:  # if data path is passed, parse_data will override data_samples
            sentences, self.ground_truths = self.parse_data(data_path)
            self.data_samples = self.unroll_data(sentences, self.ground_truths, use_pos)

        if use_pos and self.data_samples[0]["pos"] is None:
            # if we want to use pos tags but they are not precomputed, insert into each sample a new list containing the index-encoded pos tags
            # matching indexes of words in "words", e.g.: ['he', 'was'] has tags ['PRON', 'AUX'] represented as [11, 4]
            stanza.download(lang="en", processors="tokenize,pos", verbose=False)
            # note that due to this, the test container is slower: it needs to download these stanza english models each time

            pos_tagger = stanza.Pipeline(
                lang="en",
                processors="tokenize,pos",
                tokenize_pretokenized=True,
                verbose=False,
            )

            for sample in self.data_samples:
                # pos tag each sentence and collect them in the sample dictionary
                doc = pos_tagger([sample["words"]])
                sample["pos"] = [word.upos for word in doc.sentences[0].words]

        # remember to call self.index_dataset(...) if it's not a test set
        if words_vocab:  # if vocab is passed, index only the sentences directly
            self.encoded_words = []
            for i in range(len(self.words)):
                sentence = self.words[i]  # for each sentence
                self.encoded_words.append(
                    torch.LongTensor([words_vocab[token] for token in sentence])
                )  # encode it and put in object's field

    def parse_data(self, path: str) -> List[Dict]:
        """Function took from the utils.py file, reads the json dataset

        Args:
            path (str): json data file path

        Returns:
            List[Dict]: list of samples
        """

        with open(path) as f:
            dataset = json.load(f)

        sentences, labels = {}, {}
        for sentence_id, sentence in dataset.items():
            sentence_id = sentence_id
            sentences[sentence_id] = {
                "words": sentence["words"],
                "lemmas": sentence["lemmas"],
                "pos_tags": sentence["pos_tags"],
                "dependency_heads": [
                    int(head) for head in sentence["dependency_heads"]
                ],
                "dependency_relations": sentence["dependency_relations"],
                "predicates": sentence["predicates"],
            }

            labels[sentence_id] = {
                "predicates": sentence["predicates"],
                "roles": {int(p): r for p, r in sentence["roles"].items()}
                if "roles" in sentence
                else dict(),  # if "roles" is not in sentence it means that there are no predicates
            }

        return sentences, labels

    def unroll_data(self, sentences: Dict, labels: Dict, use_pos: bool) -> List[Dict]:
        """Unrolls and lowercases the sentences inside the dictionaries read by parse_data

        Args:
            sentences (Dict): maps sentence informations (words, lemmas, pos_tags, dependencies, predicates)
            labels (Dict): maps labels informations (predicates, roles)

        Returns:
            List[Dict]: list of samples
        """

        samples = []

        for id in labels:
            label_dic = labels[id]  # take the dictionary

            if label_dic["roles"]:  # if not empty
                for pred_id in label_dic["roles"]:  # for each predicate

                    predicates = ["_"] * len(label_dic["predicates"])
                    predicates[pred_id] = label_dic["predicates"][pred_id]
                    samples.append(
                        {
                            "id": id,
                            "words": sentences[id]["words"],
                            "lemmas": sentences[id]["lemmas"],
                            "pos": sentences[id]["pos_tags"] if use_pos else None,
                            "pred_id": pred_id,
                            "preds": predicates,
                            "roles": label_dic["roles"][pred_id],
                        }
                    )

            else:  # if empty
                samples.append(
                    {
                        "id": id,
                        "words": sentences[id]["words"],
                        "lemmas": sentences[id]["lemmas"],
                        "pos": sentences[id]["pos_tags"] if use_pos else None,
                        "pred_id": -1,
                        "preds": label_dic["predicates"],
                        "roles": label_dic["predicates"],  # just a list of underscores
                    }
                )

        return samples

    def index_dataset(
        self,
        words_vocabulary: Vocab,
        lemmas_vocabulary: Vocab,
        preds_vocabulary: Vocab,
        roles_vocabulary: Vocab,
    ):
        """Indexes words, lemmas, predicates, pos and roles in each data sample

        Args:
            words_vocabulary (Vocab): vocabulary of words
            lemmas_vocabulary (Vocab): vocabulary of lemmas
            preds_vocabulary (Vocab): vocabulary of predicates
            roles_vocabulary (Vocab): vocabulary of semantic roles
        """

        for sample in self.data_samples:
            # encode words only if not using transformers
            if not self.use_transformers:
                sample["words"] = torch.LongTensor(
                    [words_vocabulary[token] for token in sample["words"]]
                )
            # encode everything else into each sample dict
            sample["lemmas"] = torch.LongTensor(
                [lemmas_vocabulary[token] for token in sample["lemmas"]]
            )
            sample["preds"] = torch.LongTensor(
                [preds_vocabulary[token] for token in sample["preds"]]
            )
            sample["roles"] = torch.LongTensor(
                [
                    roles_vocabulary["attribute"]
                    # just to handle the "attriute" role typo in FR/dev.json
                    if token == "attriute" else roles_vocabulary[token]
                    for token in sample["roles"]
                ]
            )
            if sample["pos"]:  # if using pos tags
                sample["pos"] = torch.LongTensor(
                    [self.upos2i[token] for token in sample["pos"]]
                )

        self.encoded_samples = True
        return

    def __len__(self) -> int:
        if self.encoded_samples is False:
            raise RuntimeError(
                "Trying to retrieve length but index_dataset has not been invoked yet!"
            )
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        """returns a dict with idx-th encoded sentence, its pos tags and its list of labels
        Args:
            idx (int): index of sentence to retrieve
        Returns:
            Dict[str,torch.LongTensor]: a dictionary mapping every information of a sample
        """
        if self.encoded_samples is False:
            raise RuntimeError(
                "Trying to retrieve elements but index_dataset has not been invoked yet!"
            )
        return self.data_samples[idx]
