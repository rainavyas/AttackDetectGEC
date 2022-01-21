'''
Generate precision-recall curve for residue detector
'''

import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from train_residue_detector import LayerClassifier, get_embeddings
import torch
import torch.nn as nn
import sys
import os
import argparse
from gec_tools import get_sentences
from tools import get_default_device
from happytransformer import HappyTextToText

def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precisions*(beta**2))+recalls))
    f_scores = f_scores[~np.isnan(f_scores)]
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA', type=str, help='Path to original data file')
    commandLineParser.add_argument('DETECTOR', type=str, help='trained adv attack detector')
    commandLineParser.add_argument('OUT_FILE', type=str, help='.png file to save plot to')
    commandLineParser.add_argument('--attack_phrase', type=str, default='', help="universal attack phrase")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pr_residue_detector.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if args.cpu == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()
    
    # Load the GEC model
    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    
    # Load the data
    _, orig_sentences = get_sentences(args.DATA)
    adv_sentences = [t + ' ' + args.attack_phrase + '.' for t in orig_sentences]

    # Get encoder CLS embeddings
    orig_emb = get_embeddings(model, orig_sentences)
    adv_emb = get_embeddings(model, adv_sentences)

    # Load the Adv Attack Detector model
    detector = LayerClassifier(768)
    detector.load_state_dict(torch.load(args.DETECTOR, map_location=torch.device('cpu')))
    detector.eval()

    labels = np.asarray([0]*orig_emb.size(0) + [1]*adv_emb.size(0))
    X = torch.cat((orig_emb, adv_emb))

    # get predicted logits of being adversarial attack
    with torch.no_grad():
        logits = detector(X)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        adv_probs = probs[:,1].squeeze().cpu().detach().numpy()
    
    print("Got prediction probs")
    # get precision recall values and highest F1 score (with associated prec and rec)
    precision, recall, _ = precision_recall_curve(labels, adv_probs)
    best_precision, best_recall, best_f05 =  get_best_f_score(precision, recall, beta=0.5)
    print(f'Precision: {best_precision}\tRecall: {best_recall}\tF0.5: {best_f05}')

    # plot all the data
    plt.plot(recall, precision, 'r-')
    plt.plot(best_recall,best_precision,'bo')
    plt.annotate(F"F0.5={best_f05:.2f}", (best_recall,best_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(args.OUT_FILE)