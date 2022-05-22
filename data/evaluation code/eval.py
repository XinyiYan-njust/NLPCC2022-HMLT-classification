import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score,precision_score, recall_score

class CustomDataset(Dataset):
    '''Characterizes a dataset for PyTorch'''

    def __init__(self, documents):
        '''Initialization'''
        self.documents = documents

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.documents)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        return preprocess_function(self.documents.iloc[index])

def preprocess_function(docu):
    true_labels = [1 if x in docu['levels'] else 0 for x in labels_ref]
    pred_labels = [1 if x in docu['pred_labels'] else 0 for x in labels_ref]
    return torch.tensor(true_labels),torch.tensor(pred_labels)

if __name__ == '__main__':
    # samples of prediction results
    data = pd.read_json('input_sample/1652786880/results0.24.json')
    # list of level1
    level1 = pickle.load(open('./data/labels_1.rand123', 'rb'))
    # list of level2
    level2 = pickle.load(open('./data/labels_2.rand123', 'rb'))
    # list of level3
    level3 = pickle.load(open('./data/labels_3.rand123', 'rb'))
    # list of levels
    labels = pickle.load(open("./data/labels_all.rand123","rb"))

    labels_refs = [labels,level1,level2,level3]
    for labels_ref in labels_refs:
        validation_dataloader = DataLoader(CustomDataset(data), shuffle=False, batch_size=32)

        true_labels, pred_labels = [], []
        for i, batch in enumerate(validation_dataloader):
            b_labels, pred_label = batch
            true_labels.append(b_labels.numpy())
            pred_labels.append(pred_label.numpy())

        true_labels_val = [item for sublist in true_labels for item in sublist]
        pred_labels_val = [item for sublist in pred_labels for item in sublist]

        true_bools = [tl == 1 for tl in true_labels_val]
        pred_bools = [pl == 1 for pl in pred_labels_val]

        val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro')
        val_precision_accuracy = precision_score(true_bools, pred_bools, average='micro')
        val_recall_accuracy = recall_score(true_bools, pred_bools, average='micro')

        print('F1 Validation Accuracy: ', val_f1_accuracy)
        print('Precision Validation Accuracy: ', val_precision_accuracy)
        print('Recall Validation Accuracy: ', val_recall_accuracy)
        print()