'''
FGWS adapted for detection of adversarial attacks on
a GEC system.

Two variants of FGWS implemented: FGWS-prob and FGWS-edits

This script can be run in train mode to build the frequency dict or in eval mode
to apply the FGWS detector and obtain the F0.5 score
'''

import sys
import os
import argparse
from gec_tools import get_sentences, correct, return_edits
from pr_residue_detector import get_best_f_score
import json
from collections import defaultdict
import nltk
from nltk.corpus import wordnet as wn
from happytransformer import HappyTextToText, TTSettings
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
import string
import re

class FGWS():
    '''
    FGWS detector
    '''
    def __init__(self, data_path, mode, freq_dict_path, model=None, gen_args=None,
                    fgws_mode='edits', attack_phrase=None, delta=1):

        self.mode = mode
        self.freq_dict_path = freq_dict_path
        self.fgws_mode = fgws_mode
        self.model = model
        self.gen_args = gen_args
        self.attack_phrase = attack_phrase
        self.delta = delta
        _, self.orig_sentences = get_sentences(data_path)
        
    def __call__(self):
        '''
        Eval or Train FGWS detector
        '''
        if self.mode == 'train':
            self.train()
        else:
            self.eval()

    def eval(self):
        '''
        Evaluate FGWS detector
        '''
        nltk.download('omw-1.4')
        self.adv_sentences = [t + ' ' + self.attack_phrase + '.' for t in self.orig_sentences]

        with open(self.freq_dict_path) as f:
            freq_dict = json.load(f)
        self.freq_dict = defaultdict(int, freq_dict)

        # Get FGWS scores
        original_scores = []
        attack_scores = []
        for i, (o,a) in enumerate(zip(self.orig_sentences, self.adv_sentences)):
            print(f'On {i}/{len(self.orig_sentences)}')
            original_scores.append(self.get_score(o))
            attack_scores.append(self.get_score(a))

        # Calculate Best F1 score
        labels = [0]*len(original_scores) + [1]*len(attack_scores)
        scores = original_scores + attack_scores
        precision, recall, _ = precision_recall_curve(labels, scores)
        best_precision, best_recall, best_f05 =  get_best_f_score(precision, recall, beta=0.5)
        print(f'Precision: {best_precision}\tRecall: {best_recall}\tF0.5: {best_f05}')

    def train(self):
        '''
        Learn the words frequency dict
        '''
        sentences = [re.findall(r"[\w']+|[.,!?;]", s) for s in self.orig_sentences] # get list of words and punctuation
        frequencies = defaultdict(int)
        for sen in sentences:
            for word in sen:
                frequencies[word] += 1
        
        # Save dict
        with open(self.freq_dict_path, 'w') as f:
            json.dump(frequencies, f)
    
    def get_score(self, X):
        '''
        Return FGWS score for given sentence
        Identify low frequency words in sentence X
        Substitute these words with higher frequency synonyms (create X')
        Calculate change in output score from model, i.e. |f(X) - f(X')|
        '''
        word_list = re.findall(r"[\w']+|[.,!?;]", X) # get list of words and punctuation
        X_dash_words = []
        for word in word_list:
            if self.freq_dict[word] < self.delta:
                X_dash_words.append(self._substitute(word))
            else:
                X_dash_words.append(word)
        X_dash = re.sub(r' ([^A-Za-z0-9])', r'\1', ' '.join(X_dash_words)) # Want no space before punctuation

        if self.fgws_mode == 'prob':
            return self._fgws_prob(X, X_dash)
        elif self.fgws_mode == 'edits':
            return self._fgws_edits(X, X_dash)
        else:
            print('Error in FGWS mode')
    
    def _fgws_edits(self, X, X_dash):
        '''
        Calculate FGWS-edits score
        '''
        f_X = self._sent_to_edits(X)
        f_X_dash = self._sent_to_edits(X_dash)
        return abs(f_X-f_X_dash)/len(re.findall(r"[\w']+|[.,!?;]", X))
    
    def _sent_to_edits(self, sent):
        '''
        Get number of edits between input sentence and output prediction
        '''
        corr = correct(self.model, sent, self.gen_args)
        edits = return_edits(sent, corr)
        return len(edits)

    
    def _fgws_prob(self, X, X_dash):
        '''
        Calculate FGWS-prob score:
        '''
        f_X, pred_inds = self.model_pred_prob_seq(X)
        f_X_dash, _ = self.model_pred_prob_seq(X_dash, pred_inds=pred_inds)
        return f_X - f_X_dash
    
    def _substitute(self, word):
        '''
        Find synonym of word with a higher frequency
        '''
        best = (word, self.freq_dict[word])
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if self.freq_dict[lemma.name()] > best[1]:
                    best = (lemma.name(), self.freq_dict[lemma.name()])
        return best[0]
    
    def model_pred_prob_seq(self, X, pred_ins=None):
        '''
        Give model's decoded probability vector sequence output
        for input sentence X
        Returns sum of log probabilities of sequence of predicted word indices.
        Also return sequence of indices of predicted words at decoding
        '''
        # TODO: determine how to get this
    
    def remove_punctuation(sentence):
        # NO LONGER USED - instead count punctuation as separate tokens
        return sentence.translate(None, string.punctuation)



if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA', type=str, help='Path to original data file')
    commandLineParser.add_argument('--mode', type=str, choices=['train', 'eval'], help="train or eval mode")
    commandLineParser.add_argument('--freq_dict_path', type=str, help='json frequency dict filepath to save (in train) or load (in eval)')
    commandLineParser.add_argument('--attack_phrase', type=str, default='', help="universal attack phrase")
    commandLineParser.add_argument('--fgws_mode', type=str, default='edits', choices=['prob', 'edits'], help="type of FGWS variant")
    commandLineParser.add_argument('--delta', type=int, default=1, help="Frequency threshold for substitution")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/fgws.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    gen_args = TTSettings(num_beams=5, min_length=1)
    detector = FGWS(args.DATA, args.mode, args.freq_dict_path, model, gen_args,
                    args.fgws_mode, args.attack_phrase, args.delta)
    detector()

