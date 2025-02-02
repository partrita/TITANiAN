from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
import torch

from src.constants import STOP, START, MASK, PAD
from src.constants import PROTEIN_ALPHABET
from sklearn.metrics import roc_auc_score, precision_score, recall_score, r2_score
import random
def warmup(n_warmup_steps):
    def get_lr(step):
        return min((step + 1) / n_warmup_steps, 1.0)
    return get_lr


def transformer_lr(n_warmup_steps):
    factor = n_warmup_steps ** 0.5
    def get_lr(step):
        step += 1
        return min(step ** (-0.5), step * n_warmup_steps ** (-1.5)) * factor
    return get_lr


def get_metrics(fname, new=False, tokens=False):
    with open(fname) as f:
        lines = f.readlines()
    valid_lines = []
    train_lines = []
    all_train_lines = []
    for i, line in enumerate(lines):
        if 'Training' in line and 'loss' in line:
            last_train = line
            all_train_lines.append(line)
        if 'Validation complete' in line:
            valid_lines.append(lines[i - 1])
            train_lines.append(last_train)
    metrics = []
    idx_loss = 13
    idx_accu = 16
    idx_step = 6
    if new:
        idx_loss += 2
        idx_accu += 2
        idx_step += 2
    if tokens:
        idx_loss += 2
        idx_accu += 2
        idx_tok = 10
    tok_correction = 0
    last_raw_toks = 0
    for t, v in zip(train_lines, valid_lines):
        step = int(t.split()[idx_step])
        t_loss = float(t.split()[idx_loss])
        t_accu = float(t.split()[idx_accu][:6])
        v_loss = float(v.split()[idx_loss])
        v_accu = float(v.split()[idx_accu][:6])
        if tokens:
            toks = int(t.split()[idx_tok])
            if toks < last_raw_toks:
                tok_correction += last_raw_toks
                doubled = int(all_train_lines[-1].split()[idx_tok]) - int(all_train_lines[-999].split()[idx_tok])
                tok_correction -= doubled
            last_raw_toks = toks
            metrics.append((step, toks + tok_correction, t_loss, t_accu, v_loss, v_accu))

        else:
            metrics.append((step, t_loss, t_accu, v_loss, v_accu))
    if tokens:
        metrics = pd.DataFrame(metrics, columns=['step', 'tokens', 'train_loss',
                                                 'train_accu', 'valid_loss', 'valid_accu'])
    else:
        metrics = pd.DataFrame(metrics, columns=['step', 'train_loss', 'train_accu', 'valid_loss', 'valid_accu'])
    return metrics


def get_weights(seqs):
    scale = 1.0
    theta = 0.2
    seqs = np.array([[PROTEIN_ALPHABET.index(a) for a in s] for s in seqs])
    weights = scale / (np.sum(squareform(pdist(seqs, metric="hamming")) < theta, axis=0))
    return weights


def parse_fasta(fasta_fpath, return_names=False):
    """ Read in a fasta file and extract just the sequences."""
    seqs = []
    with open(fasta_fpath) as f_in:
        current = ''
        names = [f_in.readline()[1:].replace('\n', '')]
        for line in f_in:
            if line[0] == '>':
                seqs.append(current)
                current = ''
                names.append(line[1:].replace('\n', ''))
            else:
                current += line.replace('\n', '')
        seqs.append(current)
    if return_names:
        return seqs, names
    else:
        return seqs


def read_fasta(fasta_fpath, out_fpath, header='sequence'):
    """ Read in a fasta file and extract just the sequences."""
    with open(fasta_fpath) as f_in, open(out_fpath, 'w') as f_out:
        f_out.write(header + '\n')
        current = ''
        _ = f_in.readline()
        for line in f_in:
            if line[0] == '>':
                f_out.write(current + '\n')
                current = ''
            else:
                current += line[:-1]
        f_out.write(current + '\n')


class Tokenizer(object):
    """Convert between strings and their one-hot representations."""
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a:i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i:a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    @property
    def start_id(self) -> int:
        return self.alphabet.index(START)

    @property
    def stop_id(self) -> int:
        return self.alphabet.index(STOP)

    @property
    def mask_id(self) -> int:
        return self.alphabet.index(MASK)

    @property
    def pad_id(self) -> int:
        return self.alphabet.index(PAD)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x: Iterable) -> str:
        return ''.join([self.t_to_a[t] for t in x])

def one_hot_encode(seq, alphabet: int) -> np.ndarray:
    """Convert a string into a one-hot representation."""
    x,y = seq.shape
    one_hot_tensor = torch.zeros(x,y,alphabet)
    one_hot_tensor.scatter_(2,seq.unsqueeze(-1),1)
    return one_hot_tensor

def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Parameters:
    - model: the PyTorch model.

    Returns:
    - Total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_masked_accuracy(model, dataloader, device):
    """
    Calculate the accuracy of a model on masked regions.

    Parameters:
    - model: the trained model.
    - dataloader: DataLoader that feeds batches of data.
    - device: computation device (e.g., 'cuda' or 'cpu').

    Returns:
    - Accuracy of the model on the masked regions of the provided data.
    """
    model.eval()
    correct_predictions = 0
    total_masked_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, input_mask, frac = batch
            input_mask_ = input_mask.unsqueeze(-1).to(device)
            frac = torch.FloatTensor(frac).unsqueeze(-1).to(device)
            inputs, targets, input_mask,frac = inputs.to(device), targets.to(device), input_mask.to(device), frac.to(device)
            
            shared_e,task1_e,outputs= model(inputs, frac)
            _, predicted = torch.max(outputs, 2)  # Assuming outputs are [batch_size, seq_len, num_classes]
            # Consider only masked positions
            #targets = targets.argmax(dim=-1)
            masked_predicted = predicted[input_mask.bool()]
            masked_targets = targets[input_mask.bool()]

            correct_predictions += (masked_predicted == masked_targets).sum().item()
            total_masked_predictions += masked_targets.numel()

    accuracy = correct_predictions / total_masked_predictions
    return accuracy

def calculate_masked_accuracy_tri(model, dataloader, device):
    """
    Calculate the accuracy of a model on masked regions.

    Parameters:
    - model: the trained model.
    - dataloader: DataLoader that feeds batches of data.
    - device: computation device (e.g., 'cuda' or 'cpu').

    Returns:
    - Accuracy of the model on the masked regions of the provided data.
    """
    model.eval()
    correct_predictions = 0
    total_masked_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, input_mask, frac , m1, m2, t1,t2= batch
            input_mask_ = input_mask.unsqueeze(-1).to(device)
            m1, m2, t1, t2 = m1.to(device), m2.to(device), t1.to(device), t2.to(device)
            frac = torch.FloatTensor(frac).unsqueeze(-1).to(device)
            inputs, targets, input_mask,frac = inputs.to(device), targets.to(device), input_mask.to(device), frac.to(device)
            
            shared_e,task1_e,outputs= model(inputs, m1,m2,t1,t2,frac)
            _, predicted = torch.max(outputs, 2)  # Assuming outputs are [batch_size, seq_len, num_classes]
            # Consider only masked positions
            #targets = targets.argmax(dim=-1)
            masked_predicted = predicted[input_mask.bool()]
            masked_targets = targets[input_mask.bool()]

            correct_predictions += (masked_predicted == masked_targets).sum().item()
            total_masked_predictions += masked_targets.numel()

    accuracy = correct_predictions / total_masked_predictions
    return accuracy

def evaluate_tri(model, dataloader, criterion,device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (src, tgt, mask, frac, m1, m2, t1, t2) in enumerate(dataloader):
            mask = mask.unsqueeze(-1).to(device)
            frac = torch.FloatTensor(frac).unsqueeze(-1).to(device)
            m1, m2, t1, t2 = m1.to(device), m2.to(device), t1.to(device), t2.to(device)
            src, tgt, mask, frac = src.to(device), tgt.to(device), mask.to(device), frac.to(device)
            shared_e,task1_e,output = model(src, m1, m2, t1, t2, frac)
            masked_predicted = output[mask.squeeze(dim= -1).bool()]
            masked_targets = tgt[mask.squeeze(dim =-1).bool()]
            loss = criterion(masked_predicted, masked_targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion,device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (src, tgt, mask, frac) in enumerate(dataloader):
            mask = mask.unsqueeze(-1).to(device)
            frac = torch.FloatTensor(frac).unsqueeze(-1).to(device)
            src, tgt, mask, frac = src.to(device), tgt.to(device), mask.to(device), frac.to(device)
            shared_e,task1_e,output = model(src, frac)
            masked_predicted = output[mask.squeeze(dim= -1).bool()]
            masked_targets = tgt[mask.squeeze(dim =-1).bool()]
            loss = criterion(masked_predicted, masked_targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def orthogonal_loss(embedding_a, embedding_b):
    embedding_a_flat = embedding_a.reshape(embedding_a.size(0), -1)
    embedding_b_flat = embedding_b.reshape(embedding_b.size(0), -1)
    
    norm_a = torch.norm(embedding_a_flat, p=2, dim=1, keepdim=True)
    norm_b = torch.norm(embedding_b_flat, p=2, dim=1, keepdim=True)
    
    normalized_a = embedding_a_flat / norm_a
    normalized_b = embedding_b_flat / norm_b
    
    cosine_similarity = torch.mm(normalized_a, normalized_b.t())
    loss = torch.mean(torch.abs(cosine_similarity))
    
    return loss

def calculate_accuracy(y_true, y_pred):
    y_pred_class = (y_pred > 0.5).float()
    correct = (y_pred_class == y_true).float().sum()
    accuracy = correct / len(y_true)
    return accuracy.item()

# Function to calculate AUROC
def calculate_auroc(y_true, y_pred):
    y_true_binary = (y_true > 0.5).float()  # Convert to binary labels
    
    return roc_auc_score(y_true_binary.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

def calculate_precision_recall(y_true, y_pred):
    y_pred_class = (y_pred > 0.5).float()
    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    return precision, recall

def evaluate_class_2(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (seq, label) in enumerate(dataloader):
            seq = seq.to(device)
            label = torch.FloatTensor(label).unsqueeze(-1).to(device)            
            shared_e,task_e, output = model(seq)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu())
            all_targets.extend(label.detach().cpu())
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    acc = calculate_accuracy(all_targets, all_preds)
    auroc = calculate_auroc(all_targets, all_preds)
    precision, recall = calculate_precision_recall(all_targets, all_preds)

    return total_loss / len(dataloader), acc, auroc, precision, recall

def evaluate_class_tri(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (seq, label,m1,m2,t1,t2) in enumerate(dataloader):
            seq = seq.to(device)
            m1 ,m2, t1, t2 = m1.to(device), m2.to(device), t1.to(device), t2.to(device)
            label = torch.FloatTensor(label).unsqueeze(-1).to(device)
            shared_e,task_e, output = model(seq,m1, m2, t1, t2)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu())
            all_targets.extend(label.detach().cpu())

    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    acc = calculate_accuracy(all_targets, all_preds)
    auroc = calculate_auroc(all_targets, all_preds)
    precision, recall = calculate_precision_recall(all_targets, all_preds)

    return total_loss / len(dataloader), acc, auroc, precision, recall

def evaluate_class_3(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (seq, label, m1) in enumerate(dataloader):
            seq = seq.to(device)
            m1= m1.to(device)
            label = torch.FloatTensor(label).unsqueeze(-1).to(device)            
            shared_e,task_e, output = model(seq, m1)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu())
            all_targets.extend(label.detach().cpu())
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    acc = calculate_accuracy(all_targets, all_preds)
    auroc = calculate_auroc(all_targets, all_preds)
    precision, recall = calculate_precision_recall(all_targets, all_preds)

    return total_loss / len(dataloader), acc, auroc, precision, recall

def evaluate_class_4(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (seq, label, m2) in enumerate(dataloader):
            seq = seq.to(device)
            m2=  m2.to(device)
            label = torch.FloatTensor(label).unsqueeze(-1).to(device)            
            shared_e,task_e, output = model(seq, m2)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu())
            all_targets.extend(label.detach().cpu())
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    acc = calculate_accuracy(all_targets, all_preds)
    auroc = calculate_auroc(all_targets, all_preds)
    precision, recall = calculate_precision_recall(all_targets, all_preds)

    return total_loss / len(dataloader), acc, auroc, precision, recall

def evaluate_class_5(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (seq, label, t2) in enumerate(dataloader):
            seq = seq.to(device)
            t2 = t2.to(device)
            label = torch.FloatTensor(label).unsqueeze(-1).to(device)            
            shared_e,task_e, output = model(seq, t2)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu())
            all_targets.extend(label.detach().cpu())
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    acc = calculate_accuracy(all_targets, all_preds)
    auroc = calculate_auroc(all_targets, all_preds)
    precision, recall = calculate_precision_recall(all_targets, all_preds)

    return total_loss / len(dataloader), acc, auroc, precision, recall

def evaluate_class_final(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq, m1, m2, label in dataloader:
            seq = seq.to(device)
            m1, m2 = m1.to(device), m2.to(device)
            label = torch.FloatTensor(label).unsqueeze(-1).to(device)            
            _,output = model(seq,m1,m2)
            # clamp output from 0 to 1
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu())
            all_targets.extend(label.detach().cpu())
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    acc = calculate_accuracy(all_targets, all_preds)
    auroc = calculate_auroc(all_targets, all_preds)
    
    

    return total_loss / len(dataloader), acc, auroc
def evaluate_merged(model, dataloader, device,task_list=[1,2,3,4,5,6,7]):
    model.eval()
    total_loss = 0.0
    correct_predictions_1 = 0
    total_masked_predictions_1 = 0
    all_preds_task2 = []
    all_targets_task2 = []
    all_preds_task3 = []
    all_targets_task3 = []
    correct_predictions_4 = 0
    total_masked_predictions_4 = 0
    all_preds_task5 = []
    all_targets_task5 = []
    correct_predictions_6 = 0
    total_masked_predictions_6 = 0
    all_preds_task7 = []
    all_targets_task7 = []

    with torch.no_grad():
        for batch_idx, (tuple_, task) in enumerate(dataloader):
            if task[0] ==1:
                src, tgt, mask, frac = tuple_
                src, tgt = src.to(device), tgt.to(device)
                input_mask = mask.unsqueeze(-1).to(device)
                embeddings ,outputs, adv_output = model(src, frac=frac, task = task)
                output = outputs[0]
                _, predicted = torch.max(output, 2)  # Assuming outputs are [batch_size, seq_len, num_classes]
                # Consider only masked positions
                #targets = targets.argmax(dim=-1)
                masked_predicted = predicted[mask.bool()]
                masked_targets = tgt[mask.bool()]

                correct_predictions_1 += (masked_predicted == masked_targets).sum().item()
                total_masked_predictions_1 += masked_targets.numel()
            if task[0] ==2:
                src, label = tuple_
                src = src.to(device)
                label = torch.FloatTensor(label).unsqueeze(-1).to(device)
                embeddings ,outputs, adv_output = model(src,task = task)
                output = outputs[1]
                all_preds_task2.extend(output.detach().cpu())
                all_targets_task2.extend(label.detach().cpu())
            if task[0] ==3:
                src,m1,label = tuple_
                src = src.to(device)
                m1 = m1.to(device)
                label = torch.FloatTensor(label).unsqueeze(-1).to(device)
                embeddings,outputs, adv_output = model(src,m1 = m1, task = task)
                output = outputs[2]
                all_preds_task3.extend(output.detach().cpu())
                all_targets_task3.extend(label.detach().cpu())
            if task[0] ==4:
                src, tgt, mask, m1 = tuple_
                src, tgt = src.to(device), tgt.to(device)
                input_mask = mask.unsqueeze(-1).to(device)
                m1 = m1.to(device)
                embeddings ,outputs, adv_output = model(src, m1 = m1, task = task)
                output = outputs[3]
                _, predicted = torch.max(output, 2)  # Assuming outputs are [batch_size, seq_len, num_classes]
                # Consider only masked positions
                #targets = targets.argmax(dim=-1)
                masked_predicted = predicted[mask.bool()]
                masked_targets = tgt[mask.bool()]

                correct_predictions_4 += (masked_predicted == masked_targets).sum().item()
                total_masked_predictions_4 += masked_targets.numel()
            if task[0] ==5:
                src,m2,label = tuple_
                src = src.to(device)
                m2 = m2.to(device)
                label = torch.FloatTensor(label).unsqueeze(-1).to(device)
                embeddings,outputs, adv_output = model(src,m2 = m2, task = task)
                output = outputs[4]
                all_preds_task5.extend(output.detach().cpu())
                all_targets_task5.extend(label.detach().cpu())
            if task[0] ==6:
                src, tgt, mask, m2 = tuple_
                src, tgt = src.to(device), tgt.to(device)
                input_mask = mask.unsqueeze(-1).to(device)
                m2 = m2.to(device)
                embeddings ,outputs, adv_output = model(src, m2 = m2, task = task)
                output = outputs[5]
                _, predicted = torch.max(output, 2)  # Assuming outputs are [batch_size, seq_len, num_classes]
                # Consider only masked positions
                #targets = targets.argmax(dim=-1)
                masked_predicted = predicted[mask.bool()]
                masked_targets = tgt[mask.bool()]

                correct_predictions_6 += (masked_predicted == masked_targets).sum().item()
                total_masked_predictions_6 += masked_targets.numel()
            if task[0] ==7:
                src,t2, label = tuple_
                src = src.to(device)
                t2 = t2.to(device)
                label = torch.FloatTensor(label).unsqueeze(-1).to(device)
                embeddings,outputs, adv_output = model(src,t2 = t2, task = task)
                output = outputs[6]
                all_preds_task7.extend(output.detach().cpu())
                all_targets_task7.extend(label.detach().cpu())
                
    if 1 in task_list:
        accuracy_1 = correct_predictions_1 / total_masked_predictions_1
    else:
        accuracy_1 = 0
    if 2 in task_list:
        all_preds_task2 = torch.stack(all_preds_task2)
        all_targets_task2 = torch.stack(all_targets_task2)
        acc_task2 = calculate_accuracy(all_targets_task2, all_preds_task2)
        auroc_task2 = calculate_auroc(all_targets_task2, all_preds_task2)
        precision_task2, recall_task2 = calculate_precision_recall(all_targets_task2, all_preds_task2)
    else:
        acc_task2 = 0
        auroc_task2 = 0
        precision_task2 = 0
        recall_task2 = 0
    if 3 in task_list:
        all_preds_task3 = torch.stack(all_preds_task3)
        all_targets_task3 = torch.stack(all_targets_task3)
        acc_task3 = calculate_accuracy(all_targets_task3, all_preds_task3)
        auroc_task3 = calculate_auroc(all_targets_task3, all_preds_task3)
        precision_task3, recall_task3 = calculate_precision_recall(all_targets_task3, all_preds_task3)
    else:
        acc_task3 = 0
        auroc_task3 = 0
        precision_task3 = 0
        recall_task3 = 0
    if 4 in task_list:
        accuracy_4 = correct_predictions_4 / total_masked_predictions_4
    else:
        accuracy_4 = 0
    if 5 in task_list:
        all_preds_task5 = torch.stack(all_preds_task5)
        all_targets_task5 = torch.stack(all_targets_task5)
        acc_task5 = calculate_accuracy(all_targets_task5, all_preds_task5)
        auroc_task5 = calculate_auroc(all_targets_task5, all_preds_task5)
        precision_task5, recall_task5 = calculate_precision_recall(all_targets_task5, all_preds_task5)
    else:
        acc_task5 = 0
        auroc_task5 = 0
        precision_task5 = 0
        recall_task5 = 0
    if 6 in task_list:
        accuracy_6 = correct_predictions_6 / total_masked_predictions_6
    else:
        accuracy_6 = 0
    if 7 in task_list:
        all_preds_task7 = torch.stack(all_preds_task7)
        all_targets_task7 = torch.stack(all_targets_task7)
        acc_task7 = calculate_accuracy(all_targets_task7, all_preds_task7)
        auroc_task7 = calculate_auroc(all_targets_task7, all_preds_task7)
        precision_task7, recall_task7 = calculate_precision_recall(all_targets_task7, all_preds_task7)
    else:
        acc_task7 = 0
        auroc_task7 = 0
        precision_task7 = 0
        recall_task7 = 0
        
    return (accuracy_1,
            [acc_task2, auroc_task2, precision_task2 , recall_task2],
            [acc_task3, auroc_task3, precision_task3 , recall_task3],
            accuracy_4,
            [acc_task5, auroc_task5, precision_task5 , recall_task5],
            accuracy_6,
            [acc_task7, auroc_task7, precision_task7 , recall_task7])

                

# calculate R^2 score
def calculate_r2(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    return r2_score(y_true, y_pred)

def evaluate_final(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (seq, label, m1, m2, t1, t2) in enumerate(dataloader):
            seq = seq.to(device)
            m1, m2, t1, t2 = m1.to(device), m2.to(device), t1.to(device), t2.to(device)
            label = torch.FloatTensor(label).unsqueeze(-1).to(device)            
            _,output = model(seq, m1, m2, t1, t2)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.detach().cpu())
            all_targets.extend(label.detach().cpu())
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)

    r2 = calculate_r2(all_targets, all_preds)

    return total_loss / len(dataloader), r2

#Seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def calculate_ppvn_top_n(predicted_probs, actuals):
    """
    Calculate the Precision at N (PPVn) for top-N predictions based on highest probabilities.
    
    Args:
    predicted_probs (list of float): List of predicted probabilities.
    actuals (list of int): Corresponding list of actual labels.
    
    Returns:
    list: A list of PPVn values from top-1 to top-N.
    """
    # Combine predictions with actuals and sort by predicted probability in descending order
    paired = list(zip(predicted_probs, actuals))
    sorted_by_prob = sorted(paired, key=lambda x: x[0], reverse=True)
    
    ppvn_values = []
    true_positive_count = 0
    
    for n in range(1, len(sorted_by_prob) + 1):
        top_n = sorted_by_prob[:n]
        true_positive_count = sum(1 for _, actual in top_n if actual == 1)
        ppvn = true_positive_count / n
        ppvn_values.append(ppvn)
    
    return ppvn_values

def calculate_mean_ppvn_with_ci(ppv_values):
    mean_ppvn = np.mean(ppv_values)
    std_dev = np.std(ppv_values)
    z_score = norm.ppf(0.975)  # for 95% CI
    margin_error = z_score * (std_dev / np.sqrt(len(ppv_values)))
    ci_lower = mean_ppvn - margin_error
    ci_upper = mean_ppvn + margin_error

    return mean_ppvn, (ci_lower, ci_upper)