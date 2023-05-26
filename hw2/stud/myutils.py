# let's start with all the imports
# NOTE: part of this code is taken from notebook #8 - Q&A
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
from myvocab import Vocab
from typing import List, Dict

# just defining a couple of utility functions


def prepare_batch_transformers(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """collate_fn for the train and dev DataLoaders if using transformers, applies padding to data and
    takes into account the fact that [CLS] and [SEP] tokens are added from the transformer
    Args:
        batch (List[Dict]): a list of dictionaries, each dict is a sample from the Dataset
    Returns:
        Dict[str,torch.Tensor]: a batch into a dictionary {x:data, y:labels}
    """
    # extract features from batch
    ids = [sample["id"] for sample in batch]  # plain list
    words = [sample["words"] for sample in batch]  # list of lists
    lemmas = [sample["lemmas"] for sample in batch]
    pred_ids = torch.as_tensor([sample["pred_id"] for sample in batch])
    preds = [sample["preds"] for sample in batch]
    roles = [sample["roles"] for sample in batch]

    zero_col = torch.zeros((len(batch), 1), dtype=torch.int64)
    hundred_col = torch.full((len(batch), 1), -100, dtype=torch.int64)

    pos = None
    if batch[0]["pos"] is not None:  # if using pos tags
        pos = [sample["pos"] for sample in batch]
        pos = pad_sequence(
            [torch.as_tensor(sample) for sample in pos], batch_first=True
        )
        pos = torch.cat([zero_col, pos, zero_col], dim=1)

    # convert features to tensor and pad them
    lemmas = pad_sequence(
        [torch.as_tensor(sample) for sample in lemmas], batch_first=True
    )
    # add padding corresponding to [CLS] and [SEP]
    lemmas = torch.cat([zero_col, lemmas, zero_col], dim=1)

    preds = pad_sequence(
        [torch.as_tensor(sample) for sample in preds], batch_first=True
    )
    preds = torch.cat([zero_col, preds, zero_col], dim=1)

    roles = pad_sequence(
        [torch.as_tensor(sample) for sample in roles],
        batch_first=True,
        padding_value=-100,
    )
    # add -100 label corresponding to [CLS] and [SEP]
    roles = torch.cat([hundred_col, roles, hundred_col], dim=1)

    return {
        "id": ids,
        "words": words,
        "pos": pos,
        "lemmas": lemmas,
        "pred_id": pred_ids,
        "preds": preds,
        "roles": roles,
    }


def prepare_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """collate_fn for the train and dev DataLoaders, applies padding to data
    Args:
        batch (List[Dict]): a list of dictionaries, each dict is a sample from the Dataset
    Returns:
        Dict[str,torch.Tensor]: a batch into a dictionary {x:data, y:labels}
    """
    # extract features from batch
    ids = [sample["id"] for sample in batch]  # plain list
    words = [sample["words"] for sample in batch]
    lemmas = [sample["lemmas"] for sample in batch]
    pred_ids = torch.as_tensor([sample["pred_id"] for sample in batch])
    preds = [sample["preds"] for sample in batch]
    roles = [sample["roles"] for sample in batch]

    pos = None
    if batch[0]["pos"] is not None:  # if using pos tags
        pos = [sample["pos"] for sample in batch]
        pos = pad_sequence(
            [torch.as_tensor(sample) for sample in pos], batch_first=True
        )

    # convert features to tensor and pad them
    words = pad_sequence(
        [torch.as_tensor(sample) for sample in words], batch_first=True
    )
    lemmas = pad_sequence(
        [torch.as_tensor(sample) for sample in lemmas], batch_first=True
    )
    preds = pad_sequence(
        [torch.as_tensor(sample) for sample in preds], batch_first=True
    )
    roles = pad_sequence(
        [torch.as_tensor(sample) for sample in roles],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "id": ids,
        "words": words,
        "pos": pos,
        "lemmas": lemmas,
        "pred_id": pred_ids,
        "preds": preds,
        "roles": roles,
    }


def load_pretrained_embeddings(
    weights: KeyedVectors, words_vocab: Vocab, freeze: bool
) -> nn.Embedding:
    """Creates the pretrained embedding layer, according to the index mapping we have in our vocabulary
    Args:
        weights (KeyedVectors): pretrained embeddings from gensim
        words_vocab (Vocab): our vocabulary of words
        freeze (bool): whether to allow fine-tuning of pretrained embeddings or not
    Returns:
        nn.Embedding: the PyTorch embedding layer
    """
    vectors = weights.vectors
    to_be_filled = np.random.randn(
        len(words_vocab) + 1, vectors.shape[1]
    )  # +1 for padding
    to_be_filled[0] = np.zeros(vectors.shape[1])  # zero vector for padding
    to_be_filled[1] = np.mean(vectors, axis=0)  # mean vector for unknown tokens

    initialised = 0  # just for stats
    for w, i in words_vocab.w2i.items():
        if w in weights and w != "<unk>":  # if the word is in the pretrained embeddings
            initialised += 1
            vec = weights[w]
            to_be_filled[i] = vec  # insert in right position

    print("initialised embeddings: {}".format(initialised))
    print(
        "randomly initialised embeddings: {} ".format(
            len(words_vocab) - initialised - 1
        )
    )

    return nn.Embedding.from_pretrained(
        torch.FloatTensor(to_be_filled), padding_idx=0, freeze=freeze
    )
