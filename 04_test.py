import os
import sys
import pickle
import pathlib
import argparse
import numpy as np


def svmrank_process(policy, data_loader, top_k=[1, 3, 5, 10]):
  mean_kacc = np.zeros(len(top_k))
  n_samples_processed = 0

  for batch in data_loader:
    feats, best, num_cands = batch
    n_samples = best.shape[0]

    # feature normalization
    feats = (feats - policy['shift']) / policy['scale']

    # prediction
    pred_scores = policy['model'].predict(feats)
    pred_scores = torch.FloatTensor(pred_scores)
    pred_scores = pad_tensor(pred_scores, num_cands)

    # accuracy
    kacc = []
    for k in top_k:
      if pred_scores.size()[-1] < k:
        kacc.append(1.0)
        continue
      pred_top_k = pred_scores.topk(k).indices
      accuracy = (pred_top_k == best).any(dim=-1).float().mean().item()
      kacc.append(accuracy)
    kacc = np.asarray(kacc)

    mean_kacc += kacc * n_samples
    n_samples_processed += n_samples

  mean_kacc /= n_samples_processed
  return mean_kacc
    


def gnn_process(policy, data_loader, top_k=[1, 3, 5, 10]):
  mean_kacc = np.zeros(len(top_k))
  mean_entropy = 0

  n_samples_processed = 0

  for batch in data_loader:
    batch = batch.to(device)
    logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
    logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
    entropy = (-F.softmax(logits, dim=-1)*F.log_softmax(logits, dim=-1)).sum(-1).mean()

    true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
    true_bestscore = true_scores.max(dim=-1, keepdims=True).values

    kacc = []
    for k in top_k:
      if logits.size()[-1] < k:
        kacc.append(1.0)
        continue
      pred_top_k = logits.topk(k).indices
      pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
      accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
      kacc.append(accuracy)
    kacc = np.asarray(kacc)

    mean_entropy += entropy.item() * batch.num_graphs
    mean_kacc += kacc * batch.num_graphs
    n_samples_processed += batch.num_graphs

  mean_kacc /= n_samples_processed
  mean_entropy /= n_samples_processed
  return mean_kacc, mean_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
    )
    parser.add_argument(
        'model',
        help='Model to test.',
        choices=['svmrank', 'gnn'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random number generator seed.',
        type=int,
        default=0,
    )
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }
    problem_folder = problem_folders[args.problem]
    model_dir = f"model/trained_models/{args.problem}/{args.model}"

    ## logger setup ##
    logfile = os.path.join(model_dir, 'test_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    if args.model == "gnn":
      ## pytorch setup ##
      if args.gpu == -1:
          os.environ['CUDA_VISIBLE_DEVICES'] = ''
          device = "cpu"
      else:
          os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
          device = f"cuda:0"
      import torch
      import torch.nn.functional as F
      import torch_geometric
      from utilities import log, pad_tensor, GraphDataset, Scheduler
      sys.path.insert(0, os.path.abspath(f'model'))
      from model import GNNPolicy
      torch.manual_seed(args.seed)

      ## load data ##
      test_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_folder/'test').glob('sample_*.pkl')]
      test_data = GraphDataset(test_files)
      test_loader = torch_geometric.loader.DataLoader(test_data, 64, shuffle=False)

      ## load gnn model ##
      model = GNNPolicy().to(device)
      model.load_state_dict(torch.load(f"{model_dir}/train_params.pkl", map_location=device))

      ## test ##
      top_k = [1, 3, 5, 10]
      test_kacc, entropy = gnn_process(model, test_loader, top_k)
      log(f"TEST RESULTS: " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, test_kacc)]), logfile)

    else:
      import torch
      import svmrank
      from utilities import log, FlatDataset, pad_tensor

      ## load feature normalization parameters ##
      policy = {}
      with open(f"{model_dir}/normalization.pkl", 'rb') as f:
        policy['shift'], policy['scale'] = pickle.load(f)
        print(policy['shift'], policy['scale'])

      ## load data ##
      test_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_folder/'test').glob('sample_*.pkl')]
      test_data = FlatDataset(test_files)
      test_loader = torch.utils.data.DataLoader(test_data, 64, shuffle=False, collate_fn=FlatDataset.collate)

      ## load model ##
      policy['model'] = svmrank.Model().read(f"{model_dir}/model.txt")

      ## test ##
      top_k = [1, 3, 5, 10]
      test_kacc = svmrank_process(policy, test_loader)
      log(f"TEST RESULTS: " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, test_kacc)]), logfile)


      

    


    
