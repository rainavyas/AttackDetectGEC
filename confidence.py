import sys
import os
import argparse
from gec_tools import get_sentences, correct
from pr_residue_detector import get_best_f_score
from happytransformer import HappyTextToText, TTSettings
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
import math

def negative_confidence(sentence, HappyModel, gen_args, device=torch.device('cpu')):
    '''
    Calculate negative confidence of sentence using model
    '''
    sf = nn.Softmax(dim=0)
    HappyModel.model.to(device)
    model = HappyModel.model
    tokenizer = HappyModel.tokenizer
    output_sentence = correct(HappyModel, sentence, gen_args)

    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    all_decoder_input_ids = tokenizer(output_sentence, return_tensors="pt").input_ids
    all_decoder_input_ids[0, 0] = model.config.decoder_start_token_id
    assert all_decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id

    total = 0
    for i in range(1, all_decoder_input_ids.size(1)):
        decoder_input_ids = all_decoder_input_ids[:,:i]
        outputs = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
        lm_logits = outputs.logits.squeeze()
        probs = sf(lm_logits)
        pred_id = all_decoder_input_ids[:,i].squeeze().item()
        prob = probs[pred_id]
        total += math.log(prob)
    return ((-1)/all_decoder_input_ids.size(1)) * total


if __name__ == '__main__':
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA', type=str, help='Path to original data file')
    commandLineParser.add_argument('--attack_phrase', type=str, default='', help="universal attack phrase")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/confidence.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # You can change to gpt-large or other pretrained models that you can find in Huggingface.
    HappyModel = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    gen_args = TTSettings(num_beams=5, min_length=1)

    # Load the data
    _, orig_sentences = get_sentences(args.DATA)
    adv_sentences = [t + ' ' + args.attack_phrase + '.' for t in orig_sentences]

    # Get negative confidence scores
    original_scores = []
    attack_scores = []
    for i, (o,a) in enumerate(zip(orig_sentences, adv_sentences)):
        print(f'On {i}/{len(orig_sentences)}')
        # try:
        #     original_scores.append(negative_confidence(o, HappyModel, gen_args))
        #     attack_scores.append(negative_confidence(a, HappyModel, gen_args))
        # except:
        #     print("Failed for ", o)
        original_scores.append(negative_confidence(o, HappyModel, gen_args))

    labels = [0]*len(original_scores) + [1]*len(attack_scores)
    scores = original_scores + attack_scores
    precision, recall, _ = precision_recall_curve(labels, scores)
    best_precision, best_recall, best_f05 =  get_best_f_score(precision, recall, beta=0.5)
    print(f'Precision: {best_precision}\tRecall: {best_recall}\tF0.5: {best_f05}')