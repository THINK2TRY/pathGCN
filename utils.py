import argparse


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed', 'ogbn-arxiv').")
    parser.add_argument("--num_paths", type=int, default=1,)
    parser.add_argument("--path_length", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--lr_oc", type=float, default=1e-2)
    parser.add_argument("--wd_oc", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max_epochs", type=int, default=300)
    
    args = parser.parse_args()
    return args