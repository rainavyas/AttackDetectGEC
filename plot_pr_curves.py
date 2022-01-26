import sys
import os
import argparse
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from pr_residue_detector import get_best_f_score

if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('PR_FILES', type=str, help='Path to .npz files separated by space')
    commandLineParser.add_argument('NAMES', type=str, help='legend names to corresponding PR files')
    commandLineParser.add_argument('OUT', type=str, help='.png file to save to')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot_pr_curves.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load npz files
    filenames = args.PR_FILES.split()
    names = args.NAMES.split()
    assert len(filenames) == len(names)

    for f, name in zip(filenames, names):
        file_object = np.load(f)
        precision = file_object['arr_0']
        recall = file_object['arr_1']
        plt.plot(recall, precision, label=name)
        best_precision, best_recall, best_f05 =  get_best_f_score(precision, recall, beta=0.5)
        plt.plot(best_recall, best_precision, marker='x', color='k', ms=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.5,1.02)
    plt.legend()
    plt.savefig(args.OUT, bbox_inches='tight')
