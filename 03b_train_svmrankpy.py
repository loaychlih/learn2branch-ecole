import pickle
import os
import argparse
import numpy as np
import utilities
import pathlib
import svmrank

from utilities import log, load_flat_samples


def load_samples(filenames, size_limit, logfile=None):
    x, y, ncands = [], [], []
    total_ncands = 0

    for i, filename in enumerate(filenames):
        cand_x, cand_y, best = load_flat_samples(filename)

        x.append(cand_x)
        y.append(cand_y)
        ncands.append(cand_x.shape[0])
        total_ncands += ncands[-1]

        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(filenames)} files processed ({total_ncands} candidate variables)", logfile)

        if total_ncands >= size_limit:
            log(f"  dataset size limit reached ({size_limit} candidate variables)", logfile)
            break

    x = np.concatenate(x)
    y = np.concatenate(y)
    ncands = np.asarray(ncands)

    if total_ncands > size_limit:
        x = x[:size_limit]
        y = y[:size_limit]
        ncands[-1] -= total_ncands - size_limit

    return x, y, ncands


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'problem',
      help='MILP instance type to process.',
      choices=['setcover', 'cauctions', 'facilities', 'indset'],
  )
  parser.add_argument(
      '-s', '--seed',
      help='Random generator seed.',
      type=utilities.valid_seed,
      default=0,
  )
  args = parser.parse_args()
  rng = np.random.RandomState(args.seed)

  # config #
  train_max_size = 250000
  valid_max_size = 100000
  feat_type = 'khalil'
  feat_qbnorm = True
  feat_augment = True
  label_type = 'bipartite_ranks'

  # input/output directories #
  problem_folders = {
      'setcover': 'setcover/500r_1000c_0.05d',
      'cauctions': 'cauctions/100_500',
      'facilities': 'facilities/100_100_5',
      'indset': 'indset/500_4',
  }
  problem_folder = problem_folders[args.problem]
  running_dir = f"trained_models/{args.problem}/svmrank/{args.seed}"
  os.makedirs(running_dir, exist_ok=True)

  # logging configuration #
  logfile = f"{running_dir}/log.txt"
  log(f"Logfile for svmrank on {args.problem} with seed {args.seed}", logfile)

  # data loading #
  train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
  valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))
  log(f"{len(train_files)} training files", logfile)
  log(f"{len(valid_files)} validation files", logfile)

  log("Loading training samples", logfile)
  train_x, train_y, train_ncands = load_samples(rng.permutation(train_files), train_max_size, logfile)
  log(f"  {train_x.shape[0]} training samples", logfile)

  log("Loading validation samples", logfile)
  valid_x, valid_y, valid_ncands = load_samples(valid_files, valid_max_size, logfile)
  log(f"  {valid_x.shape[0]} validation samples", logfile)

  # data normalization
  log("Normalizing datasets", logfile)
  x_shift = train_x.mean(axis=0)
  x_scale = train_x.std(axis=0)
  x_scale[x_scale == 0] = 1
  valid_x = (valid_x - x_shift) / x_scale
  train_x = (train_x - x_shift) / x_scale

  # save normalization parameters
  with open(f"{running_dir}/normalization.pkl", "wb") as f:
      pickle.dump((x_shift, x_scale), f)

  log("Starting training", logfile)

  train_qids = np.repeat(np.arange(len(train_ncands)), train_ncands)
  valid_qids = np.repeat(np.arange(len(valid_ncands)), valid_ncands)

  # Training (includes hyper-parameter tuning)
  best_loss = np.inf
  best_model = None
  for c in (1e-3, 1e-2, 1e-1, 1e0):
      log(f"C: {c}", logfile)
      model = svmrank.Model({
          '-c': c * len(train_ncands),  # c_light = c_rank / n
          '-v': 1,
          '-y': 0,
          '-l': 2,
      })
      print(train_x.shape)
      print(train_y.shape)
      print(train_qids.shape)
      model.fit(train_x, train_y, train_qids)
      loss = model.loss(train_y, model(train_x, train_qids), train_qids)
      log(f"  training loss: {loss}", logfile)
      loss = model.loss(valid_y, model(valid_x, valid_qids), valid_qids)
      log(f"  validation loss: {loss}", logfile)
      if loss < best_loss:
          best_model = model
          best_loss = loss
          best_c = c
          # save model
          model.write(f"{running_dir}/model.txt")

  log(f"Best model with C={best_c}, validation loss: {best_loss}", logfile)