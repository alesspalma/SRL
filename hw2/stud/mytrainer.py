# let's start with all the imports
# NOTE: part of this code is taken from notebook #5
import torch
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import transformers_embedder as tre
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from myvocab import Vocab
from typing import Dict


class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(self, model:nn.Module, loss_function, optimizer, label_vocab:Vocab, device):
        """Constructor of our trainer
        Args:
            model (nn.Module): model to train
            loss_function (nn.Loss): loss function to use
            optimizer (nn.Optim): optimizer to use
            label_vocab (Vocab): label vocabulary used to decode the output
            device (torch.device): device where to perform training and validation
        """
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = label_vocab
        self.tokenizer = tre.Tokenizer(model.use_transformer, add_prefix_space=True) if model.use_transformer is not None else None
        
        self.predictions = [] # will contain validation set predictions, useful to plot confusion matrix
        self.truths = [] # will contain validation set truths, useful to plot confusion matrix

    def train(self, train_data:DataLoader, valid_data:DataLoader, epochs:int, patience:int, path:str) -> Dict[str, list]:
        """Train and validate the model using early stopping with patience for the given number of epochs
        Args:
            train_data (DataLoader): a DataLoader instance containing the training dataset
            valid_data (DataLoader): a DataLoader instance used to evaluate learning progress
            epochs: the number of times to iterate over train_data
            patience (int): patience for early stopping
            path (str): path where to save weights of best epoch
        Returns:
            Dict[str, list]: dictionary containing mappings { metric:value }
        """

        train_history = []
        valid_loss_history = []
        valid_argid_f1_history = []
        valid_argclass_f1_history = []
        patience_counter = 0
        best_f1 = 0.0

        print('Training on', self.device, 'device')
        print('Start training ...')
        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train() # put model in train mode

            for batch in tqdm(train_data, leave=False):
                if self.tokenizer:
                    batch['words'] = self.tokenizer(batch['words'], padding=True, return_tensors=True, is_split_into_words=True).to(self.device)
                else:
                    batch['words'] = batch['words'].to(self.device)
                batch['lemmas'], batch['preds'], y_data = batch['lemmas'].to(self.device), batch['preds'].to(self.device), batch['roles'].to(self.device)
                if batch["pos"] is not None: batch["pos"] = batch["pos"].to(self.device) # if using pos, move to device

                self.optimizer.zero_grad()
                logits = self.model(batch) # forward step output has shape: batchsize, max_seq, 27 classes

                if self.model.crf is not None: # if using crf, then use its loss computation to optimize the model
                    mask = (y_data != -100)
                    batch_losses = self.model.crf.forward(logits, y_data.where(mask, torch.LongTensor([0]).to(self.device)), mask)
                    # I mapped the -100 "padding" labels to 0 just because TorchCRF doesn't want indices out of the
                    # number of classes, but anyway they will be ignored thanks to the mask parameter
                    sample_loss = - batch_losses.mean()
                else: # otherwise, use standard given loss function on logits
                    sample_loss = self.loss_function(logits.permute(0,2,1), y_data) # permute to match nn.CrossEntropyLoss input dims
                
                sample_loss.backward() # backpropagation
                self.optimizer.step() # optimize parameters

                epoch_loss += sample_loss.item() * len(batch["id"]) # avg batch loss * precise number of batch elements

            avg_epoch_loss = epoch_loss / len(train_data.dataset) # total loss / number of samples = average sample loss for this epoch
            train_history.append(avg_epoch_loss)
            print('  [E:{:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_metrics = self.evaluate(valid_data) # validation step
            valid_loss_history.append(valid_metrics["loss"])
            valid_argid_f1_history.append(valid_metrics["argument_identification"]["f1"])
            valid_argclass_f1_history.append(valid_metrics["argument_classification"]["f1"])
            print('\t[E:{:2d}] Arg Ident valid f1 = {:0.4f}%, Arg Class valid f1 = {:0.4f}%'.format(epoch, 
                                                                                    valid_metrics["argument_identification"]["f1"]*100, 
                                                                                    valid_metrics["argument_classification"]["f1"]*100))

            # save model if the validation metric is the best ever
            if valid_metrics["argument_classification"]["f1"] > best_f1: 
                best_f1 = valid_metrics["argument_classification"]["f1"]
                torch.save(self.model.state_dict(), path)

            stop = epoch > 0 and valid_argclass_f1_history[-1] < valid_argclass_f1_history[-2]  # check if early stopping
            if stop:
                patience_counter += 1
                if patience_counter > patience: # in case we exhausted the patience, we stop
                    print('\tEarly stop\n')
                    break
                else:
                    print('\t-- Patience')
            print()

        print('Done!')
        return {
            "train_history": train_history,
            "valid_loss_history": valid_loss_history,
            "valid_argid_f1_history": valid_argid_f1_history,
            "valid_argclass_f1_history": valid_argclass_f1_history
        }

    def evaluate(self, valid_data:DataLoader) -> Dict[str, float]:
        """ perform validation of the model
        Args:
            valid_dataset: the DataLoader to use to evaluate the model.
        Returns:
            Dict[str, float]: dictionary containing mappings { metric:value }
        """
        valid_loss = 0.0
        self.predictions = [] # reset predictions and truths lists
        self.truths = []
        dockerlike_output = {} # group the samples with same id, like in the original dataset

        self.model.eval() # inference mode
        with torch.no_grad():
            for batch in tqdm(valid_data, leave=False):
                if self.tokenizer:
                    batch['words'] = self.tokenizer(batch['words'], padding=True, return_tensors=True, is_split_into_words=True).to(self.device)
                else:
                    batch['words'] = batch['words'].to(self.device)
                batch['lemmas'], batch['preds'], y_data = batch['lemmas'].to(self.device), batch['preds'].to(self.device), batch['roles'].to(self.device)
                if batch["pos"] is not None: batch["pos"] = batch["pos"].to(self.device) # if using pos, move to device

                batch_size: int = len(batch["id"])
                logits = self.model(batch)

                if self.model.crf is not None: # if using crf, then use its loss computation to evaluate the model
                    mask = (y_data != -100)
                    batch_losses = self.model.crf.forward(logits, y_data.where(mask, torch.LongTensor([0]).to(self.device)), mask)
                    sample_loss = - batch_losses.mean()
                    predictions = self.model.crf.viterbi_decode(logits, mask)

                else: # otherwise, use standard given loss function on logits
                    sample_loss = self.loss_function(logits.permute(0,2,1), y_data) # permute to match CrossEntropyLoss input dim
                    predictions = torch.argmax(logits, -1)

                valid_loss += sample_loss.item() * batch_size # avg batch loss * precise number of batch elements

                for batch_id in range(batch_size): # for each sample in this batch
                    
                    single_prediction = predictions[batch_id].view(-1) # take one sample and flatten it
                    mask = batch["lemmas"][batch_id] != 0  # remove padding
                    single_prediction = single_prediction[mask].tolist()
                    single_prediction = [self.label_vocab.i2w[idx] for idx in single_prediction]  # convert from indices to labels

                    sample_id = batch["id"][batch_id] # sample id
                    pred_idx = batch["pred_id"][batch_id].item() # predicate index
                    
                    # append to global variables
                    self.predictions.append(single_prediction)
                    single_gt = y_data[batch_id].view(-1)
                    single_gt = single_gt[mask].tolist()
                    self.truths.append([self.label_vocab.i2w[idx] for idx in single_gt])
                    
                    # now we fill the docker-like output dict
                    if pred_idx == -1: # if there are no predicates
                        dockerlike_output[sample_id] = {"roles": dict()}
                    else:
                        if sample_id in dockerlike_output:
                            dockerlike_output[sample_id]["roles"][pred_idx] = single_prediction
                        else:
                            dockerlike_output[sample_id] = {"roles": {pred_idx: single_prediction}}
        
        return {
            "loss": valid_loss / len(valid_data.dataset), # total loss / number of samples = average sample loss for validation step
            "argument_identification": self.evaluate_argument_identification(valid_data.dataset.ground_truths, dockerlike_output),
            "argument_classification": self.evaluate_argument_classification(valid_data.dataset.ground_truths, dockerlike_output)
        }

    def evaluate_argument_classification(self, labels:Dict, predictions:Dict, null_tag="_") -> Dict:
        """Function taken from the utils.py file, to evaluate the F1 score of argument classification

        Args:
            labels (Dict): ground truth
            predictions (Dict): our predictions
            null_tag (str, optional): tag for the "no role" class. Defaults to "_".

        Returns:
            Dict: a dict with metrics
        """

        true_positives, false_positives, false_negatives = 0, 0, 0
        for sentence_id in labels:
            gold = labels[sentence_id]["roles"]
            pred = predictions[sentence_id]["roles"]
            predicate_indices = set(gold.keys()).union(pred.keys())

            for idx in predicate_indices:
                if idx in gold and idx not in pred:
                    false_negatives += sum(1 for role in gold[idx] if role != null_tag)
                elif idx in pred and idx not in gold:
                    false_positives += sum(1 for role in pred[idx] if role != null_tag)
                else:  # idx in both gold and pred
                    for r_g, r_p in zip(gold[idx], pred[idx]):
                        if r_g != null_tag and r_p != null_tag:
                            if r_g == r_p:
                                true_positives += 1
                            else:
                                false_positives += 1
                                false_negatives += 1
                        elif r_g != null_tag and r_p == null_tag:
                            false_negatives += 1
                        elif r_g == null_tag and r_p != null_tag:
                            false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_argument_identification(self, labels:Dict, predictions:Dict, null_tag="_") -> Dict:
        """Function taken from the utils.py file, to evaluate the F1 score of argument identification

        Args:
            labels (Dict): ground truth
            predictions (Dict): our predictions
            null_tag (str, optional): tag for the "no role" class. Defaults to "_".

        Returns:
            Dict: a dict with metrics
        """

        true_positives, false_positives, false_negatives = 0, 0, 0
        for sentence_id in labels:
            gold = labels[sentence_id]["roles"]
            pred = predictions[sentence_id]["roles"]
            predicate_indices = set(gold.keys()).union(pred.keys())

            for idx in predicate_indices:
                if idx in gold and idx not in pred:
                    false_negatives += sum(1 for role in gold[idx] if role != null_tag)
                elif idx in pred and idx not in gold:
                    false_positives += sum(1 for role in pred[idx] if role != null_tag)
                else:  # idx in both gold and pred
                    for r_g, r_p in zip(gold[idx], pred[idx]):
                        if r_g != null_tag and r_p != null_tag:
                            true_positives += 1
                        elif r_g != null_tag and r_p == null_tag:
                            false_negatives += 1
                        elif r_g == null_tag and r_p != null_tag:
                            false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
 
    def generate_cm(self, path:str):
        """save to image the confusion matrix of the validation set of this trainer
        Args:
            path (str): path where to save the image
        """

        labels = ["_", "agent", "co-agent", "theme", "co-theme", "patient", "co-patient", "beneficiary", "topic", "goal",
                "recipient", "result", "stimulus", "experiencer", "destination", "value", "attribute", "location",
                "source", "cause", "time", "product", "purpose", "instrument", "extent", "asset", "material"]
        cm = np.around(confusion_matrix([label for sentence in self.truths for label in sentence],
                                        [label for sentence in self.predictions for label in sentence],
                                        labels=labels,
                                        normalize="true"), # normalize over ground truths
                        decimals=2) 

        df_cm = pd.DataFrame(cm, index=labels, columns=labels) # create a dataframe just for easy plotting with seaborn
        plt.figure(figsize = (30,30))
        cm_plot = sn.heatmap(df_cm, annot=True, fmt='g')
        cm_plot.set_xlabel('Predicted labels') # add some interpretability
        cm_plot.set_ylabel('True labels')
        cm_plot.set_title('Confusion Matrix')
        cm_plot.figure.savefig(path, bbox_inches='tight', pad_inches=1)
        return

    @staticmethod
    def plot_logs(logs:Dict[str, list], path:str):
        """Utility function to generate plot for metrics of loss in train vs validation. Code taken from notebook #5
        Args:
            logs (Dict[str, list]): dictionary containing the metrics
            path (str): path of the image to be saved
        """
        plt.figure(figsize=(8,6)) # create the figure

        # plot losses over epochs
        plt.plot(list(range(len(logs['train_history']))), logs['train_history'], label='Train loss')
        plt.plot(list(range(len(logs['valid_loss_history']))), logs['valid_loss_history'], label='Validation loss')

        # add some labels
        plt.title("Train vs Validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(path, bbox_inches='tight')
        return

    