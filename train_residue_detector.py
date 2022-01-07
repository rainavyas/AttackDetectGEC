'''
Train a simple linear classifier in encoder embedding space
to be able to distinguish between original and adversarial samples.
'''

import sys
import os
import argparse
import torch.nn as nn
from tools import AverageMeter, get_default_device, accuracy_topk
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from happytransformer import HappyTextToText, TTSettings
from gec_tools import get_sentences

def get_embeddings(model, sentences, device=torch.device('cpu')):
    '''
    Map input sentences to encoder embbedding space
    Use CLS token encoder embedding
    '''
    encoder = model.model.get_encoder().to(device)
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for sentence in sentences:
            input_ids = model.tokenizer(sentence, return_tensors="pt").input_ids
            encoder_outputs = encoder(input_ids)
            hidden_states = encoder_outputs[0]
            CLS_embedding = hidden_states[0, 0, :].squeeze()
            embeddings.append(CLS_embedding)
        all = torch.stack(embeddings, dim=0)
    return all


class LayerClassifier(nn.Module):
    '''
    Simple Linear classifier
    '''
    def __init__(self, dim, classes=2):
        super().__init__()
        self.layer = nn.Linear(dim, classes)
    def forward(self, X):
        return self.layer(X)

def train(train_loader, model, criterion, optimizer, epoch, device, out_file, print_freq=1):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            text = '\n Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, prec=accs)
            print(text)
            with open(out_file, 'a') as f:
                f.write(text)

def eval(val_loader, model, criterion, device, out_file):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (x, target) in enumerate(val_loader):

            x = x.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

    text ='\n Test\t Loss ({loss.avg:.4f})\t Accuracy ({prec.avg:.3f})\n'.format(loss=losses, prec=accs)
    print(text)
    with open(out_file, 'a') as f:
        f.write(text)


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA', type=str, help='Path to original data file')
    commandLineParser.add_argument('OUT', type=str, help='file to print results to')
    commandLineParser.add_argument('CLASSIFIER_OUT', type=str, help='.th to save linear adv attack classifier to')
    commandLineParser.add_argument('--attack_phrase', type=str, default='', help="universal attack phrase")
    commandLineParser.add_argument('--num_points', type=int, default=2600, help="number of data points to use")
    commandLineParser.add_argument('--num_points_val', type=int, default=800, help="number of data points to use for validation")
    commandLineParser.add_argument('--N', type=int, default=3, help="Num word substitutions used in attack")
    commandLineParser.add_argument('--B', type=int, default=100, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify learning rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")
    args = commandLineParser.parse_args()

    torch.manual_seed(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_residue_detector.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get device
    if args.cpu == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()
    
    # Load the model
    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    
    # Load the data
    _, orig_sentences = get_sentences(args.DATA, args.num_points)
    adv_sentences = [t + ' ' + args.attack_phrase for t in orig_sentences]

    # Get encoder CLS embeddings
    orig_emb = get_embeddings(model, orig_sentences)
    adv_emb = get_embeddings(model, adv_sentences)

    labels = torch.LongTensor([0]*orig_emb.size(0)+[1]*adv_emb.size(0))
    X = torch.cat((orig_emb, adv_emb))

    # Shuffle all the data
    indices = torch.randperm(len(labels))
    labels = labels[indices]
    X = X[indices]

    # Split data
    X_val = X[:args.num_points_val]
    labels_val = labels[:args.num_points_val]
    X_train = X[args.num_points_val:]
    labels_train = labels[args.num_points_val:]

    ds_train = TensorDataset(X_train, labels_train)
    ds_val = TensorDataset(X_val, labels_val)
    dl_train = DataLoader(ds_train, batch_size=args.B, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.B)

    # Model
    detector = LayerClassifier(X.size(-1))
    detector.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Create file
    with open(args.OUT, 'w') as f:
        text = f'N {args.N}\n'
        f.write(text)

    # Train
    for epoch in range(args.epochs):

        # train for one epoch
        text = '\n current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        with open(args.OUT, 'a') as f:
            f.write(text)
        print(text)
        train(dl_train, detector, criterion, optimizer, epoch, device, args.OUT)

        # evaluate
        eval(dl_val, detector, criterion, device, args.OUT)
    
    # Save the trained model for identifying adversarial attacks
    torch.save(detector.state_dict(), args.CLASSIFIER_OUT)