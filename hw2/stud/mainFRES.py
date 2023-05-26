# this script was executed to train the models, let's start with all the imports
import os
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from mydataset import SRLDataset
from myvocab import Vocab
from mytrainer import Trainer
from mymodel import ModHParams, SRLModel
from myutils import prepare_batch_transformers, prepare_batch

# fix the seed to allow reproducibility
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# data paths, please note that this script ran on my local computer
MODEL_FOLDER = "model/"
LANGUAGE = "FR"  # FR or ES
DATA_FOLDER = "data/" + LANGUAGE + "/"
TRAIN_DATA = DATA_FOLDER + "train.json"
VAL_DATA = DATA_FOLDER + "dev.json"

# if using transformers or not
USE_TRANSFORMERS = True
if USE_TRANSFORMERS:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# if transfer learning, initialize weights of the model from the english pretrained model
TRANSFER_LEARNING = True
TRANSFER_LEARNING_CLASSIFIER = True  # whether to transfer also the classification layer

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate datasets and vocabs
train_data = SRLDataset(TRAIN_DATA, use_pos=True, use_transformers=USE_TRANSFORMERS)

if not os.path.exists(MODEL_FOLDER + "words_vocab" + LANGUAGE + ".json"):
    words_vocab, lemmas_vocab, preds_vocab, _ = Vocab.build_vocabs(
        train_data.data_samples, return_also_roles=False
    )  # instantiate all the vocabs from the train set
    words_vocab.dump(MODEL_FOLDER + "words_vocab" + LANGUAGE + ".json")  # save them
    lemmas_vocab.dump(MODEL_FOLDER + "lemmas_vocab" + LANGUAGE + ".json")
    preds_vocab.dump(MODEL_FOLDER + "preds_vocab" + LANGUAGE + ".json")
else:  # just load them
    words_vocab = Vocab.load(MODEL_FOLDER + "words_vocab" + LANGUAGE + ".json")
    lemmas_vocab = Vocab.load(MODEL_FOLDER + "lemmas_vocab" + LANGUAGE + ".json")
    preds_vocab = Vocab.load(MODEL_FOLDER + "preds_vocab" + LANGUAGE + ".json")
# the roles vocab must be in common, we want the same n of classes
roles_vocab = Vocab.load(MODEL_FOLDER + "roles_vocab.json")

if TRANSFER_LEARNING:
    # if using transfer learning, also the predicates must be in common with the english model
    preds_vocab = Vocab.load(MODEL_FOLDER + "preds_vocabEN.json")

# encodes the dataset with the vocabularies
train_data.index_dataset(words_vocab, lemmas_vocab, preds_vocab, roles_vocab)

val_data = SRLDataset(VAL_DATA, use_pos=True, use_transformers=USE_TRANSFORMERS)
val_data.index_dataset(words_vocab, lemmas_vocab, preds_vocab, roles_vocab)

# create DataLoaders
workers = min(os.cpu_count(), 4)
train_dataloader = DataLoader(
    train_data,
    batch_size=3,
    collate_fn=prepare_batch_transformers if USE_TRANSFORMERS else prepare_batch,
    num_workers=workers,
    shuffle=True,
)

valid_dataloader = DataLoader(
    val_data,
    batch_size=3,
    collate_fn=prepare_batch_transformers if USE_TRANSFORMERS else prepare_batch,
    num_workers=workers,
    shuffle=False,
)

# create model hyperparameters, note that the default values of ModHParams constructor are already the best-found hyperparameters
params = ModHParams(
    words_vocab,
    preds_vocab,
    lemmas_vocab,
    roles_vocab,
    use_transformer="camembert-base"
    if LANGUAGE == "FR"
    else "PlanTL-GOB-ES/roberta-base-bne",
)

# instantiate and train the model
srlmodel = SRLModel(params)
learning_rate = 0.001

if TRANSFER_LEARNING:
    en_weigths = torch.load(MODEL_FOLDER + "best_weightsEN.pt")
    en_weigths = {
        k: v for k, v in en_weigths.items() if "transformer" not in k
    }  # remove transformer parameters
    if not TRANSFER_LEARNING_CLASSIFIER:
        en_weigths = {
            k: v for k, v in en_weigths.items() if "classifier" not in k
        }  # remove final classifier parameters
    srlmodel.load_state_dict(en_weigths, strict=False)
    learning_rate = 1e-3  # 1e-4, 1e-5

# optimizer initialization
optimizer = torch.optim.Adam(
    srlmodel.parameters(), lr=learning_rate, weight_decay=1e-5
)  # weight decay = L2 regularization

trainer = Trainer(
    model=srlmodel,
    loss_function=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    label_vocab=roles_vocab,
    device=device,
)

metrics = trainer.train(
    train_dataloader,
    valid_dataloader,
    40,
    9,
    MODEL_FOLDER + "best_weights" + LANGUAGE + ".pt",
)
print(
    "BEST ARG CLASS VALID F1: {:0.4f}% AT EPOCH {}".format(
        max(metrics["valid_argclass_f1_history"]) * 100,
        np.argmax(metrics["valid_argclass_f1_history"]) + 1,
    )
)
trainer.generate_cm("hw2/stud/cm.png")  # confusion matrix
Trainer.plot_logs(metrics, "hw2/stud/losses.png")  # loss plot
