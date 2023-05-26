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

# if true, execute the grid search on hyperparameters, else just train the manually-set model at the end of this script
GRID_SEARCH = False

# fix the seed to allow reproducibility
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# data paths, please note that this script ran on my local computer
MODEL_FOLDER = "model/"
LANGUAGE = "EN"
DATA_FOLDER = "data/" + LANGUAGE + "/"
TRAIN_DATA = DATA_FOLDER + "train.json"
VAL_DATA = DATA_FOLDER + "dev.json"

# if using transformers or not
USE_TRANSFORMERS = True
if USE_TRANSFORMERS:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate datasets and vocabs
train_data = SRLDataset(TRAIN_DATA, use_pos=True, use_transformers=USE_TRANSFORMERS)

if not os.path.exists(MODEL_FOLDER + "words_vocab" + LANGUAGE + ".json"):
    words_vocab, lemmas_vocab, preds_vocab, roles_vocab = Vocab.build_vocabs(
        train_data.data_samples
    )  # instantiate all the vocabs from the train set
    words_vocab.dump(MODEL_FOLDER + "words_vocab" + LANGUAGE + ".json")  # save them
    lemmas_vocab.dump(MODEL_FOLDER + "lemmas_vocab" + LANGUAGE + ".json")
    preds_vocab.dump(MODEL_FOLDER + "preds_vocab" + LANGUAGE + ".json")
    roles_vocab.dump(MODEL_FOLDER + "roles_vocab.json")
else:  # just load them
    words_vocab = Vocab.load(MODEL_FOLDER + "words_vocab" + LANGUAGE + ".json")
    lemmas_vocab = Vocab.load(MODEL_FOLDER + "lemmas_vocab" + LANGUAGE + ".json")
    preds_vocab = Vocab.load(MODEL_FOLDER + "preds_vocab" + LANGUAGE + ".json")
    roles_vocab = Vocab.load(MODEL_FOLDER + "roles_vocab.json")

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

if GRID_SEARCH:
    # grid search on some hyperparameters
    for pred_dim in [150, 300]:
        for pos_dim in [50, 100]:
            for hidden in [256, 512]:
                for lrate in [0.0005, 0.001]:
                    for wdecay in [0, 1e-5]:

                        # create model hyperparameters
                        params = ModHParams(
                            words_vocab,
                            preds_vocab,
                            lemmas_vocab,
                            roles_vocab,
                            use_transformer="roberta-base",
                            predicates_embedding_dim=pred_dim,
                            hidden_dim=hidden,
                            pos_embedding_dim=pos_dim,
                        )

                        # instantiate and train the model
                        srlmodel = SRLModel(params)
                        trainer = Trainer(
                            model=srlmodel,
                            loss_function=nn.CrossEntropyLoss(),
                            optimizer=torch.optim.Adam(
                                srlmodel.parameters(),
                                lr=lrate,
                                weight_decay=wdecay,
                            ),  # weight decay = L2 regularization
                            label_vocab=roles_vocab,
                            device=device,
                        )

                        metrics = trainer.train(
                            train_dataloader,
                            valid_dataloader,
                            60,
                            9,
                            MODEL_FOLDER + "dump.pt",
                        )
                        with open("hw2/stud/grid_search_results.txt", "a") as f:
                            f.write(
                                "PRED {}, POS {}, HIDDEN {}, LR {}, DECAY {}: BEST VALID F1: {:0.3f}% AT EPOCH {}, TRAINED FOR {} EPOCHS\n".format(
                                    pred_dim,
                                    pos_dim,
                                    hidden,
                                    lrate,
                                    wdecay,
                                    max(metrics["valid_argclass_f1_history"]) * 100,
                                    np.argmax(metrics["valid_argclass_f1_history"]) + 1,
                                    len(metrics["train_history"]),
                                )
                            )
else:
    # after this coarse-grained hyperparameter tuning, I can find the best candidate model by looking in the grid_search_results.txt file
    # After finding it, I re-train it to save the weights and plot the confusion matrix as well the loss during training and validation.

    # create model hyperparameters, note that the default values of ModHParams constructor are already the best-found hyperparameters
    params = ModHParams(
        words_vocab,
        preds_vocab,
        lemmas_vocab,
        roles_vocab,
        use_transformer="roberta-base",
    )

    # instantiate the model
    srlmodel = SRLModel(params)

    # optimizer initialization
    groups = [
        {"params": srlmodel.predicates_embedding.parameters()},
        {"params": srlmodel.lstm.parameters()},
        {"params": srlmodel.fc.parameters()},
        {"params": srlmodel.classifier.parameters()},
        {
            "params": srlmodel.word_embedding.parameters(),
            "lr": 1e-5,  # 3e-5, 1e-5
            "weight_decay": 0.0,
        },
    ]
    if params.lemma_embedding_dim != 0:  # if using lemmas
        groups.append({"params": srlmodel.lemma_embedding.parameters()})
    if params.pos_embedding_dim != 0:  # if using pos
        groups.append({"params": srlmodel.pos_embedding.parameters()})

    optimizer = (
        torch.optim.Adam(groups, lr=0.001)
        if params.fine_tune_trans
        else torch.optim.Adam(srlmodel.parameters(), lr=0.001)
    )  # weight decay = L2 regularization

    # train
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
