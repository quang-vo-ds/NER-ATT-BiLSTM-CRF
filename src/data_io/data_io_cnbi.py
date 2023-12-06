from datasets import load_dataset
from src.utils.utils import get_words_num

class DataIONCBI():
    """
    DataIONCBI is an input/output data wrapper for NCBI dataset.
    """
    def __init__(self, dataset_name='ncbi_disease', train_no=None, dev_no=None, test_no=None):
        self.NCBIDataset = load_dataset(dataset_name)
        self.train_no = train_no
        self.dev_no = dev_no
        self.test_no = test_no
    
    def read_train_dev_test(self, args):
        word_sequences_train, tag_sequences_train = self.read_data('train', verbose=args.verbose, exp_no=self.train_no)
        word_sequences_dev, tag_sequences_dev = self.read_data('validation', verbose=args.verbose, exp_no=self.dev_no)
        word_sequences_test, tag_sequences_test = self.read_data('test', verbose=args.verbose, exp_no=self.test_no)
        return word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, tag_sequences_test

    def read_data(self, mode, verbose=True, exp_no=None):
        dataset = self.NCBIDataset[mode]
        word_sequences = list()
        tag_sequences = list()
        for i, row in enumerate(dataset):
            if len(row['tokens']) == 0 or len(row['ner_tags']) == 0:
                continue
            word_sequences.append(row['tokens'])
            tag_sequences.append(row['ner_tags'])
            if exp_no:
                if i>= exp_no-1:
                    break
            
        if verbose:
            print('Loading from %s: %d samples, %d words.' % (mode, len(word_sequences), get_words_num(word_sequences)))
        return word_sequences, tag_sequences