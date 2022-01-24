import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import sys
import os
import argparse
from gec_tools import get_sentences
from pr_residue_detector import get_best_f_score
from sklearn.metrics import precision_recall_curve
# from statistics import mean



def perplexity(sentence:str, tokenizer, model, stride:int=512) -> float:
    encodings = tokenizer(sentence, return_tensors='pt').input_ids
    max_length = model.config.n_positions  # 1024
    lls = []
    for i in range(0, encodings.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings[:,begin_loc:end_loc]
        target_ids = input_ids.clone()

        target_ids[:,:-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len
        
        lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    # print(ppl)
    return ppl.item()

if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA', type=str, help='Path to original data file')
    commandLineParser.add_argument('PR', type=str, help='.npz file to save precision recall values')
    commandLineParser.add_argument('--attack_phrase', type=str, default='', help="universal attack phrase")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/perplexity.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # You can change to gpt-large or other pretrained models that you can find in Huggingface.
    tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')

    # Load the data
    _, orig_sentences = get_sentences(args.DATA)
    adv_sentences = [t + ' ' + args.attack_phrase + '.' for t in orig_sentences]

    # Get perplexity scores
    original_scores = []
    attack_scores = []
    for i, (o,a) in enumerate(zip(orig_sentences, adv_sentences)):
        print(f'On {i}/{len(orig_sentences)}')
        try:
            original_scores.append(perplexity(o, tokenizer, model))
            attack_scores.append(perplexity(a, tokenizer, model))
        except:
            print("Failed for ", o)

    # original_scores = [5000 if type(s)!=float else s for s in original_scores]
    # attack_scores = [5000 if type(s)!=float else s for s in attack_scores]
    # print(mean(original_scores))
    # print(mean(attack_scores))
    # Calculate Best F score
    labels = [0]*len(original_scores) + [1]*len(attack_scores)
    scores = original_scores + attack_scores
    scores = [s if (s<5000 and type(s)==float) else 5000 for s in scores]
    # print(scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    best_precision, best_recall, best_f05 =  get_best_f_score(precision, recall, beta=0.5)
    print(f'Precision: {best_precision}\tRecall: {best_recall}\tF0.5: {best_f05}')

    # Save the pr data
    np.savez(args.PR, precision, recall)