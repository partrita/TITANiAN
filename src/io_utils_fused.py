from typing import List, Any, Iterable
import random

import numpy as np
import torch
import torch.nn.functional as F
from src.constants import PAD, GAP, START, STOP, MASK, MSA_PAD, PROTEIN_ALPHABET,ALL_AAS
from src.utils_el import Tokenizer

from torch.utils.data import Dataset,WeightedRandomSampler
import pandas as pd
from torch.utils.data import Sampler
import csv

from src.utils_el import one_hot_encode

def _pad(tokenized: List[torch.Tensor], value: int) -> torch.Tensor:
    """Utility function that pads batches to the same length."""
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    output = torch.zeros((batch_size, max_len), dtype=tokenized[0].dtype) + value
    for row, t in enumerate(tokenized):
        output[row, :len(t)] = t
    return output

class SimpleCollater(object):
    """A collater that pads and possibly reverses batches of sequences.

    Parameters:
        alphabet (str)
        pad (Boolean)
        backwards (Boolean)

    If sequences are reversed, the padding is still on the right!

    Input (list): a batch of sequences as strings
    Output (torch.LongTensor): tokenized batch of sequences
    """

    def __init__(self, alphabet: str, pad=False, backwards=False, pad_token=PAD):
        self.pad = pad
        #self.tokenizer = AAIndexTokenizer(dpath='/home/jasonkjh/works/projects/immunogenicity/data/task1/')
        self.tokenizer = Tokenizer(alphabet)
        self.backwards = backwards
        #self.pad_idx = self.tokenizer.alphabet.index(pad_token)

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        fraction = data[1]
        prepped = self._prep(sequences,fraction)
        return prepped

    def _prep(self, sequences):
        if self.backwards:
            sequences = [s[::-1] for s in sequences]
        sequences = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]
        if self.pad:
            sequences = _pad(sequences, self.pad_idx)
        else:
            sequences = torch.stack(sequences)
        return (sequences,)
    
class CSVDataset_finetune(Dataset):
    def __init__(self,data:pd.DataFrame):
        self.seq = data['peptide']
        self.pseudo = data['pseudo']
        self.label = data['label']
        self.task = data['task']
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self,idx):
        peptide = self.seq[idx]
        pseudo = self.pseudo[idx]
        label = self.label[idx]
        task = self.task[idx]
        
        if task ==1:
            return peptide, PAD, PAD, label, task
        elif task ==3:
            return peptide, pseudo, PAD, label,task
    
class Collater_finetune(SimpleCollater):
    def __init__(self, alphabet: str, task_list ,pad=False, backwards=False, pad_token=PAD, mut_alphabet=ALL_AAS):
        super().__init__(alphabet, pad=pad, backwards=backwards, pad_token=pad_token)
        self.mut_alphabet=mut_alphabet
        self.task_list = task_list
        
    def _prep_task1(self, sequences, mhc1, mhc2, labels,task):
        seq_ = list(sequences[:])
        seq = []
        label = list(labels[:])
        mhc1_ = list(mhc1[:])
        mhc1 = []
        mhc2_ = list(mhc2[:])
        mhc2 = []
        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            m1 = mhc1_[i]
            while len(m1) < 50:
                m1 += PAD
            mhc1.append(m1)
            m2 = mhc2_[i]
            while len(m2) < 50:
                m2 += PAD
            mhc2.append(m2)
            

        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc1]
        m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc2]
        seq = torch.stack(seq)
        m1 = torch.stack(m1)
        m2 = torch.stack(m2)
        seq = one_hot_encode(seq,29)
        m1 = one_hot_encode(m1,29)
        m2 = one_hot_encode(m2,29)

        return seq, m1, m2, label, task
    
    def __call__(self, batch):
        sequences, mhc1, mhc2, fractions, task = zip(*batch)
        return self._prep_task1(sequences, mhc1, mhc2, fractions,task)
            
    
class CSVDataset_inf_pep(Dataset):
    def __init__(self, data:pd.DataFrame):
        self.seq = data['peptide']

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self,idx):
        peptide = self.seq[idx]
        return peptide, PAD, PAD
    
class CSVDataset_test(Dataset):
    def __init__(self, data:pd.DataFrame):
        self.seq = data['peptide']
        self.mhc1 = data['pseudo']
        self.mhc2 = data['pseudo']
        self.tcr = data['CDR3b']
        self.label = data['label']
        self.mhc_type = data['mhc']

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self,idx):
        peptide = self.seq[idx]
        m1 = self.mhc1[idx]
        m2 = self.mhc2[idx]
        tcr = self.tcr[idx]
        label = self.label[idx]
        peptide_length = len(peptide)
        mhc_type = self.mhc_type[idx]
        return peptide, m1, m2, tcr, label, peptide_length, mhc_type
    
class Collater_inf_pep(SimpleCollater):
    def __init__(self, alphabet: str, pad=False, backwards=False, pad_token=PAD, mut_alphabet=ALL_AAS):
        super().__init__(alphabet, pad=pad, backwards=backwards, pad_token=pad_token)
        self.mut_alphabet=mut_alphabet
    
    def _prep(self, sequences, mhc1, mhc2):
        seq_ = list(sequences[:])
        seq = []
        mhc1_ = list(mhc1[:])
        mhc1 = []
        mhc2_ = list(mhc2[:])
        mhc2 = []
        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            m1 = mhc1_[i]
            while len(m1) < 50:
                m1 += PAD
            mhc1.append(m1)
            m2 = mhc2_[i]
            while len(m2) < 50:
                m2 += PAD
            mhc2.append(m2)
            

        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc1]
        m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc2]
        seq = torch.stack(seq)
        m1 = torch.stack(m1)
        m2 = torch.stack(m2)
        seq = one_hot_encode(seq,29)
        m1 = one_hot_encode(m1,29)
        m2 = one_hot_encode(m2,29)

        return seq, m1, m2
    def __call__(self, batch):
        sequences, m1, m2 = zip(*batch)
        
        return self._prep(sequences, m1, m2)
    
class Collater_test(SimpleCollater):
    def __init__(self, alphabet: str, pad=False, backwards=False, pad_token=PAD, mut_alphabet=ALL_AAS):
        super().__init__(alphabet, pad=pad, backwards=backwards, pad_token=pad_token)
        self.mut_alphabet=mut_alphabet
    
    def _prep(self, sequences, mhc1, mhc2,tcr, labels, peptide_length, mhc_type):
        seq_ = list(sequences[:])
        seq = []
        mhc1_ = list(mhc1[:])
        mhc1 = []
        mhc2_ = list(mhc2[:])
        mhc2 = []
        tcr_ = list(tcr[:])
        tcr = []
        label = list(labels[:])
        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            m1 = mhc1_[i]
            while len(m1) < 50:
                m1 += PAD
            mhc1.append(m1)
            m2 = mhc2_[i]
            while len(m2) < 50:
                m2 += PAD
            mhc2.append(m2)
            t = tcr_[i]
            while len(t) < 25:
                t += PAD
            tcr.append(t)
            

        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc1]
        m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc2]
        tcr = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tcr]
        seq = torch.stack(seq)
        m1 = torch.stack(m1)
        m2 = torch.stack(m2)
        tcr = torch.stack(tcr)
        seq = one_hot_encode(seq,29)
        m1 = one_hot_encode(m1,29)
        m2 = one_hot_encode(m2,29)
        tcr = one_hot_encode(tcr,29)
        return seq, m1, m2, tcr, label, peptide_length, mhc_type
    def __call__(self, batch):
        sequences, m1, m2,tcr, label, peptide_length, mhc_type = zip(*batch)
        return self._prep(sequences, m1, m2,tcr, label, peptide_length, mhc_type)


class StreamedCSVDataSet(Dataset):
    def __init__(self, csv_file_path, index_file_path):
        self.csv_file_path = csv_file_path
        self.index = self._load_index(index_file_path)
        self.num_lines = self._get_num_lines()
    
    def _get_num_lines(self):
        with open(self.csv_file_path, 'r') as file:
            return sum(1 for line in file)
    
    def __len__(self):
        # Subtract 1 for the header
        return self.num_lines - 1 
        
    def _load_index(self, index_file_path):
        with open(index_file_path, 'r') as file:
            return [int(line.strip()) for line in file]
    
    def __getitem__(self, idx):
        position = self.index[idx+1]
        with open(self.csv_file_path, 'r') as file:
            file.seek(position)
            line = file.readline()
            row = next(csv.reader([line]))
            
            # Extract the required fields
            peptide = row[0]
            pseudo = row[1]
            CDR3b = row[2]
            label = float(row[3])
            task = int(row[4])
            
            # Your logic here
            # Assuming PAD is a predefined constant
            if task == 1 or task == 2:
                return peptide, PAD, PAD, PAD, label, task
            elif task in [3, 4, 8]:
                return peptide, pseudo, PAD, PAD, label, task
            elif task == 5 or task == 6:
                return peptide, PAD, pseudo, PAD, label, task
            elif task == 7:
                return peptide, PAD, PAD, CDR3b, label, task

class merged_csv_Dataset(Dataset):
    def __init__(self,data:pd.DataFrame):
        self.seq = data['peptide']
        self.pseudo = data['pseudo']
        self.CDR3b = data['CDR3b']
        self.label = data['label']
        self.task = data['task']
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self,idx):
        peptide = self.seq[idx]
        pseudo = self.pseudo[idx]
        CDR3b = self.CDR3b[idx]
        label = self.label[idx]
        task = self.task[idx]

        if task == 1:
            return peptide, PAD, PAD, PAD, label, task
        elif task == 2:
            return peptide, PAD, PAD, PAD, label, task
        elif task == 3:
            return peptide, pseudo, PAD, PAD, label, task
        elif task == 4:
            return peptide, pseudo, PAD, PAD, label, task
        elif task == 5:
            return peptide, pseudo, PAD, PAD, label, task
        elif task == 6:
            return peptide, PAD, pseudo, PAD, label, task
        elif task == 7:
            return peptide, PAD, pseudo, PAD, label, task
        elif task == 8:
            return peptide, pseudo, PAD, PAD, label, task
        elif task == 9:
            return peptide, PAD, PAD, CDR3b, label, task

from torch.utils.data import Sampler, DataLoader
import numpy as np

class TaskSpecificBatchSampler_ablation(Sampler):
    def __init__(self, data_source, batch_size, task_list):
        self.data_source = data_source
        self.batch_size = batch_size
        self.task_list = task_list
        self.task_indices = self._group_indices_by_task()
        self.batches = self._prepare_batches()

    def _group_indices_by_task(self):
        task_indices = {}
        for idx in range(len(self.data_source)):
            task = self.data_source[idx][-1]  # Assuming task is the last item
            if task not in task_indices and task in self.task_list:
                task_indices[task] = []
            if task in self.task_list:
                task_indices[task].append(idx)
        return task_indices

    def _prepare_batches(self):
        batches = []
        for indices in self.task_indices.values():
            np.random.shuffle(indices)  # Shuffle indices within each task
            batched_indices = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            batches.extend(batched_indices)
        np.random.shuffle(batches)  # Shuffle batches for randomness
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    
class TaskSpecificBatchSampler_sampling(Sampler):
    def __init__(self, data_source, batch_size, task_fraction_dict):
        """
        Initializes the sampler.
        
        Parameters:
        - data_source: The dataset to sample from.
        - batch_size: The size of each batch.
        - task_fraction_dict: A dictionary mapping each task to the fraction of data to be sampled from that task.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.task_fraction_dict = task_fraction_dict
        self.sampled_indices = None

    def _sample_indices_by_task(self):
        """
        Samples indices for each task based on the specified fraction.
        
        Returns:
        A dictionary of sampled indices for each task.
        """
        if self.sampled_indices is not None:
            # If already sampled in this epoch, return the cached results
            return self.sampled_indices
        
        task_indices = {}
        for idx in range(len(self.data_source)):
            task = self.data_source[idx][-1]  # Assuming task is the last item
            if task not in task_indices:
                task_indices[task] = []
            task_indices[task].append(idx)
        
        sampled_indices = {}
        for task, indices in task_indices.items():
            fraction = self.task_fraction_dict.get(task, 1)  # Default fraction is 1 if not specified
            sample_size = int(len(indices) * fraction)
            sampled_indices[task] = np.random.choice(indices, sample_size, replace=False).tolist()
        
        self.sampled_indices = sampled_indices  # Cache the results for this epoch
        return sampled_indices

    def _prepare_batches(self):
        """
        Prepares batches from the sampled indices.
        
        Returns:
        A list of batches, where each batch contains indices for the data points to be included.
        """
        sampled_indices = self._sample_indices_by_task()  # Ensure indices are sampled
        batches = []
        for indices in sampled_indices.values():
            np.random.shuffle(indices)
            batched_indices = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            batches.extend(batched_indices)
        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        """
        Iterates over the batches for one epoch, dynamically sampling the dataset for each epoch.
        """
        batches = self._prepare_batches()
        for batch in batches:
            yield batch
        self.sampled_indices = None  # Reset for the next epoch

    def __len__(self):
        """
        Returns the total number of batches. This implementation may vary in the number of batches per epoch
        due to the dynamic sampling of the dataset.
        """
        if self.sampled_indices is None:
            self._sample_indices_by_task()  # Ensure indices are sampled for length calculation
        total_sample_size = sum(len(indices) for indices in self.sampled_indices.values())
        return (total_sample_size + self.batch_size - 1) // self.batch_size
    
class TaskSpecificBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.task_indices = self._group_indices_by_task()
        self.batches = self._prepare_batches()

    def _group_indices_by_task(self):
        task_indices = {}
        for idx in range(len(self.data_source)):
            task = self.data_source[idx][-1]  # Assuming task is the last item
            if task not in task_indices:
                task_indices[task] = []
            task_indices[task].append(idx)
        return task_indices

    def _prepare_batches(self):
        batches = []
        for indices in self.task_indices.values():
            np.random.shuffle(indices)  # Shuffle indices within each task
            batched_indices = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            batches.extend(batched_indices)
        np.random.shuffle(batches)  # Shuffle batches for randomness
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    
class Collater_merged(SimpleCollater):
    def __init__(self, alphabet: str, task_list,pad=False, backwards=False, pad_token=PAD, mut_alphabet=ALL_AAS):
        super().__init__(alphabet, pad=pad, backwards=backwards, pad_token=pad_token)
        self.mut_alphabet=mut_alphabet
        self.task_list = task_list

    def _prep_task1(self, sequences, fractions,task):
        tgt_ = list(sequences[:])
        tgt = []
        frac = list(fractions[:])
        src = []
        mask = []
        for i,seq in enumerate(sequences):
            tgt1 = tgt_[i]
            while len(tgt1) <20:
                tgt1 += PAD
            tgt.append(tgt1)
            if len(seq) == 0:
                tgt.remove(seq)
                continue
            mod_idx = random.sample(list(range(len(seq))), int(len(seq) * 0.15))
            if len(mod_idx) == 0:
                mod_idx = [np.random.choice(len(seq))]  # make sure at least one aa is chosen
            seq_mod = list(seq)
            for idx in mod_idx:
                p = np.random.uniform()
                if p <= 0.10:  # do nothing
                    mod = seq[idx]
                elif 0.10 < p <= 0.20:  # replace with random amino acid
                    mod = np.random.choice([i for i in self.mut_alphabet if i != seq[idx]])
                else:  # mask
                    mod = MASK
                seq_mod[idx] = mod
            while len(seq_mod) < 20:
                seq_mod += PAD
            src.append(''.join(seq_mod))
            m = torch.zeros(len(seq_mod))
            m[mod_idx] = 1
            mask.append(m)
        src = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in src]
        tgt = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tgt]
        src = torch.stack(src)
        tgt = torch.stack(tgt)
        #tgt = one_hot_encode(tgt,29)
        mask = _pad(mask, 0)
        src = one_hot_encode(src,29)

        
        return (src, tgt, mask, frac),task
    
    def _prep_task2(self, sequences, labels,task):
        seq_ = list(sequences[:])
        seq = []
        label = list(labels[:])
        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)

        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        seq = torch.stack(seq)
        
        seq = one_hot_encode(seq,29)

        return  (seq, label) ,task

    def _prep_task3(self, sequences, mhc, labels,task):
        seq_ = list(sequences[:])
        seq = []
        label = list(labels[:])
        mhc1_ = list(mhc[:])
        mhc1 = []
        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            m1 = mhc1_[i]
            while len(m1) < 50:
                m1 += PAD
            mhc1.append(m1)



        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc1]
        
        seq = torch.stack(seq)
        m1 = torch.stack(m1)
        seq = one_hot_encode(seq,29)
        m1 = one_hot_encode(m1,29)

        return  (seq,m1,label),task
    
    def _prep_task4(self, sequences, mhc, labels,task):
        seq_ = list(sequences[:])
        seq = []
        label = list(labels[:])
        mhc1_ = list(mhc[:])
        mhc1 = []
        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            m1 = mhc1_[i]
            while len(m1) < 50:
                m1 += PAD
            mhc1.append(m1)



        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc1]
        
        seq = torch.stack(seq)
        m1 = torch.stack(m1)
        seq = one_hot_encode(seq,29)
        m1 = one_hot_encode(m1,29)

        return  (seq,m1,label),task
    
    def _prep_task5(self, sequences, mhc, task):
        tgt_ = list(sequences[:])
        tgt = []
        src = []
        mask = []
        mhc1_ = list(mhc[:])
        mhc1 = []
        for i,seq in enumerate(sequences):
            tgt1 = tgt_[i]
            while len(tgt1) <20:
                tgt1 += PAD
            tgt.append(tgt1)
            if len(seq) == 0:
                tgt.remove(seq)
                continue
            mod_idx = random.sample(list(range(len(seq))), int(len(seq) * 0.15))
            if len(mod_idx) == 0:
                mod_idx = [np.random.choice(len(seq))]  # make sure at least one aa is chosen
            seq_mod = list(seq)
            for idx in mod_idx:
                p = np.random.uniform()
                if p <= 0.10:  # do nothing
                    mod = seq[idx]
                elif 0.10 < p <= 0.20:  # replace with random amino acid
                    mod = np.random.choice([i for i in self.mut_alphabet if i != seq[idx]])
                else:  # mask
                    mod = MASK
                seq_mod[idx] = mod
            while len(seq_mod) < 20:
                seq_mod += PAD
            src.append(''.join(seq_mod))
            m = torch.zeros(len(seq_mod))
            m[mod_idx] = 1
            mask.append(m)
            m1 = mhc1_[i]
            while len(m1) < 50:
                m1 += PAD
            mhc1.append(m1)
        src = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in src]
        tgt = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tgt]
        m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc1]
        src = torch.stack(src)
        tgt = torch.stack(tgt)
        m1 = torch.stack(m1)
        
        #tgt = one_hot_encode(tgt,29)
        mask = _pad(mask, 0)
        src = one_hot_encode(src,29)
        m1 = one_hot_encode(m1,29)
        
        return (src, tgt, mask, m1),task

    def _prep_task6(self, sequences, mhc, labels,task):
        seq_ = list(sequences[:])
        seq = []
        label = list(labels[:])
        mhc2_ = list(mhc[:])
        mhc2 = []

        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            m2 = mhc2_[i]
            while len(m2) < 50:
                m2 += PAD
            
            mhc2.append(m2)


        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc2]
        
        seq = torch.stack(seq)
        m2 = torch.stack(m2)
        seq = one_hot_encode(seq,29)
        m2 = one_hot_encode(m2,29)

        return (seq, m2, label),task
    
    def _prep_task7(self, sequences, mhc, labels,task):
        seq_ = list(sequences[:])
        seq = []
        label = list(labels[:])
        mhc2_ = list(mhc[:])
        mhc2 = []

        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            m2 = mhc2_[i]
            while len(m2) < 50:
                m2 += PAD
            
            mhc2.append(m2)


        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc2]
        
        seq = torch.stack(seq)
        m2 = torch.stack(m2)
        seq = one_hot_encode(seq,29)
        m2 = one_hot_encode(m2,29)

        return (seq, m2, label),task
    
    def _prep_task8(self, sequences, mhc, task):
        tgt_ = list(sequences[:])
        tgt = []
        src = []
        mask = []
        mhc2_ = list(mhc[:])
        mhc2 = []
        for i,seq in enumerate(sequences):
            tgt1 = tgt_[i]
            while len(tgt1) <20:
                tgt1 += PAD
            tgt.append(tgt1)
            if len(seq) == 0:
                tgt.remove(seq)
                continue
            mod_idx = random.sample(list(range(len(seq))), int(len(seq) * 0.15))
            if len(mod_idx) == 0:
                mod_idx = [np.random.choice(len(seq))]  # make sure at least one aa is chosen
            seq_mod = list(seq)
            for idx in mod_idx:
                p = np.random.uniform()
                if p <= 0.10:  # do nothing
                    mod = seq[idx]
                elif 0.10 < p <= 0.20:  # replace with random amino acid
                    mod = np.random.choice([i for i in self.mut_alphabet if i != seq[idx]])
                else:  # mask
                    mod = MASK
                seq_mod[idx] = mod
            while len(seq_mod) < 20:
                seq_mod += PAD
            src.append(''.join(seq_mod))
            m = torch.zeros(len(seq_mod))
            m[mod_idx] = 1
            mask.append(m)
            m2 = mhc2_[i]
            while len(m2) < 50:
                m2 += PAD
            mhc2.append(m2)
        src = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in src]
        tgt = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tgt]
        m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in mhc2]
        src = torch.stack(src)
        tgt = torch.stack(tgt)
        m2 = torch.stack(m2)
        
        #tgt = one_hot_encode(tgt,29)
        mask = _pad(mask, 0)
        src = one_hot_encode(src,29)
        m2 = one_hot_encode(m2,29)
        
        return (src, tgt, mask, m2),task
    
    def _prep_task9(self, sequences, tcr, labels,task):
        seq_ = list(sequences[:])
        seq = []
        label = list(labels[:])
        tcr2_ = list(tcr[:])
        tcr2 = []
        for i,_ in enumerate(sequences):
            seq1 = seq_[i]
            while len(seq1) < 20:
                seq1 += PAD
            seq.append(seq1)
            t2 = tcr2_[i]
            while len(t2) < 25:
                t2 += PAD
            m1 = PAD*50
            m2 = PAD*50
            t1 = PAD*20
            
            tcr2.append(t2)
        seq = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in seq]
        t2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tcr2]

        seq = torch.stack(seq)
        t2 = torch.stack(t2)
        
        seq = one_hot_encode(seq,29)
        t2 = one_hot_encode(t2,29)

        return (seq, t2, label),task
    
    def __call__(self, batch):
        peptide, pseudo1, pseudo2, CDR3b, label, task = zip(*batch)
        if task[0] in self.task_list:
            if task[0] == 1:
                return self._prep_task1(peptide, label, task)
            elif task[0] == 2:
                return self._prep_task2(peptide, label,task)
            elif task[0] == 3:
                return self._prep_task3(peptide, pseudo1, label, task)
            elif task[0] == 4:
                return self._prep_task4(peptide, pseudo1, label, task)
            elif task[0] == 5:
                return self._prep_task5(peptide, pseudo1, task)
            elif task[0] == 6:
                return self._prep_task6(peptide, pseudo2, label,task)
            elif task[0] == 7:
                return self._prep_task7(peptide, pseudo2, label,task)
            elif task[0] == 8:
                return self._prep_task8(peptide, pseudo2, task)
            elif task[0] == 9:
                return self._prep_task9(peptide, CDR3b, label,task)
        else:
            pass