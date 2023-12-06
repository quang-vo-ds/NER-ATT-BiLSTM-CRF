from __future__ import print_function

import os, sys
current_working_dir = os.getcwd()
sys.path.append(current_working_dir)

from src.factories.factory_tagger import *
from src.models.tagger_birnn_cnn_crf import *
from src.seq_indexers.seq_indexer_tag import *
from src.seq_indexers.seq_indexer_word import *
from src.seq_indexers.seq_indexer_char import *
from src.utils.utils import get_test_sequence

class TaggerPredictor():
    def __init__(self, gpu, load):
        self.gpu = gpu
        self.load = load
        # Load tagger model
        self.tagger = TaggerFactory.load(self.load, self.gpu)
        self.labels = {0: "no disease",
                       1: "first disease token",
                       2: "subsequent disease tokens"}
        
    def predict(self, input_sentence):
        word_sequences = get_test_sequence(input_sentence)
        output_tag_sequence_num = self.tagger.predict_tags_from_words(word_sequences, batch_size=1)[0]
        output_tag_sequence = [self.labels[i] for i in output_tag_sequence_num]
        return list(zip(word_sequences[0], output_tag_sequence))

if __name__ == "__main__":
    predictor = TaggerPredictor(gpu = -1,
                                load = 'src/pretrained/2023_12_02_02-57_22_tagger.hdf5'
                                )
    input_sentence = "Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia."
    print(predictor.predict(input_sentence))