import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_dataset, get_random_walk_path
from utils import build_args
from model import PathGCN


def evaluate(features, paths, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(features, paths)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    

def train(features, paths, labels, masks, model, optimizer, max_epochs):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    loss_fcn = nn.CrossEntropyLoss()

    best_model = None
    best_val_acc = 0
    test_acc = 0
    #training loop        
    for epoch in range(max_epochs):
        model.train()
        logits = model(features, paths)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(features, paths, labels, val_mask, model)
        if acc > best_val_acc:
            best_val_acc = acc
            best_model = copy.deepcopy(model)
            test_acc = evaluate(features, paths, labels, test_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | ValAccuracy {:.4f}, TestAcc: {:.4f} "
              . format(epoch, loss.item(), acc, test_acc))
    return best_model


if __name__ == '__main__':
    args = build_args()
    print(args)
    dataset = load_dataset(args.dataset)    
    
    g, (num_features, num_classes) = dataset
    g = g.remove_self_loop().add_self_loop()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.int().to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
    
    in_size = num_features
    out_size = num_classes
    model = PathGCN(
        in_dim=in_size, hidden_dim=args.hidden_dim, out_dim=out_size, dropout=args.dropout,
        num_layers=args.num_layers, num_paths=args.num_paths, path_length=args.path_length,
    )
    model = model.to(device)
    optimizer = model.setup_optimizer(args.lr, args.wd, args.lr_oc, args.wd_oc)
    paths = get_random_walk_path(g, args.num_paths, args.path_length-1)
    paths = paths.to(device).long()

    # model training
    print('Training...')
    best_model = train(features, paths, labels, masks, model, optimizer, args.max_epochs)

    # test the model
    print('Testing...')
    acc = evaluate(features, paths, labels, masks[2], best_model)
    print("Test accuracy {:.4f}".format(acc))
