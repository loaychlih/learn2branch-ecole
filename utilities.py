import gzip
import pickle
import datetime
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric

def valid_seed(seed):
  """Check whether seed is a valid random seed or not."""
  seed = int(seed)
  if seed < 0 or seed > 2**32 - 1:
    raise argparse.ArgumentTypeError(
          "seed must be any integer between 0 and 2**32 - 1 inclusive")
  return seed

def log(str, logfile=None):
  str = f'[{datetime.datetime.now()}] {str}'
  print(str)
  if logfile is not None:
    with open(logfile, mode='a') as f:
      print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
  max_pad_size = pad_sizes.max()
  output = input_.split(pad_sizes.cpu().numpy().tolist())
  output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                        for slice_ in output], dim=0)
  return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, _, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph


class FlatDataset(torch.utils.data.Dataset):
  def __init__(self, filenames):
    self.filenames = filenames

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    with gzip.open(self.filenames[idx], 'rb') as file:
      sample = pickle.load(file)

    _, khalil_state, best_cand, cands, scores = sample['data']

    cands = np.array(cands)
    cand_scores = np.array(scores[cands])
    cand_states = np.array(khalil_state[:,:-24]) # TO DO!! Fix nan bug !!
    best_cand_idx = np.where(cands == best_cand)[0][0]

    # add interactions to state
    interactions = (
        np.expand_dims(cand_states, axis=-1) * \
        np.expand_dims(cand_states, axis=-2)
    ).reshape((cand_states.shape[0], -1))
    cand_states = np.concatenate([cand_states, interactions], axis=1)

    # normalize to state
    cand_states -= cand_states.min(axis=0, keepdims=True)
    max_val = cand_states.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1
    cand_states /= max_val

    # scores quantile discretization as in
    cand_labels = np.empty(len(cand_scores), dtype=int)
    cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
    cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0

    return cand_states, cand_labels, best_cand_idx

  def collate(batch):
    num_candidates = [item[0].shape[0] for item in batch]
    num_candidates = torch.LongTensor(num_candidates)

    # batch states #
    batched_states = [item[0] for item in batch]
    batched_states = np.concatenate(batched_states, axis=0)
    # batch targets #
    batched_best = [[item[2]] for item in batch]
    batched_best = torch.LongTensor(batched_best)

    return [batched_states, batched_best, num_candidates]

class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


