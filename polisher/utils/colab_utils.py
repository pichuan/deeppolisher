# Copyright (c) 2023, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Utilities for Colab exploration/visualization of Deeploid/Deepolisher.

Example usage in this notebook: ../colab_utils_notebook.ipynb
"""

from typing import Any, Optional, Union

import ml_collections
import numpy as np
import tensorflow as tf

from polisher.make_images import encoding
from polisher.models import model_utils

vocab = ''.join(encoding.get_vocab())
vocab_lookup = np.vectorize(vocab.__getitem__)


def initialize_model(
    checkpoint_path: str,
    params: ml_collections.ConfigDict,
    example_shape: tuple[int, int, int, int],
) -> tuple[Any, ml_collections.ConfigDict]:
  """Initializes the model and gathers parameters."""
  model_utils.modify_params(
      params=params,
      speedy=True,
      max_length=params.max_length,
      is_training=False,
  )

  model = model_utils.get_model(params)
  checkpoint = tf.train.Checkpoint(model=model)
  model_utils.print_model_summary(model, example_shape)
  checkpoint.restore(
      checkpoint_path
  ).expect_partial().assert_existing_objects_matched()

  return model, params


def load_model_from_checkpoint(
    checkpoint_path: str,
    example_shape: tuple[int, int, int, int] = (32, 121, 100, 1),
) -> tuple[Any, ml_collections.ConfigDict]:
  """Loads the model from a checkpoint.

  Args:
    checkpoint_path: Path to the checkpoint.
    example_shape: (batch, height including reads, width in bp, 1), where batch
      doesn't matter. Only the 2nd and 3rd values are used. Example: (32, 121,
      100, 1)

  Returns:
    An initialized model, the params of the model
  """
  params = model_utils.read_params_from_json(checkpoint_path=checkpoint_path)
  loaded_model, _ = initialize_model(
      checkpoint_path=checkpoint_path,
      params=params,
      example_shape=example_shape,
  )
  return loaded_model, params


def break_example_into_feature_rows(
    example: np.ndarray, max_coverage: int = 30
) -> dict[str, Any]:
  """Break the rows of an example into component features."""
  feature_rows = {
      'reference': 1,
      'encoded_bases': max_coverage,
      'encoded_match_mismatch': max_coverage,
      'encoded_base_qualities': max_coverage,
      'encoded_mapping_quality': max_coverage,
  }
  # Sets slices indicating rows for each feature type.
  feature_indices = dict()
  i_rows = 0
  for k, v in feature_rows.items():
    feature_indices[k] = slice(i_rows, i_rows + feature_rows[k])
    i_rows += v
  features = {}
  for feature in feature_rows:
    row_slice = feature_indices[feature]
    rows = example[row_slice]
    features[feature] = rows[:, :, 0]
  return features


FEATURE_TO_STRING = {
    'reference': {0: '-', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: '*'},
    'encoded_bases': {0: '-', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: '*'},
    'encoded_match_mismatch': {0: '-', 1: '.', 2: 'X'},
}


def decode_to_string(row: np.ndarray, feature: str) -> str:
  decoded_row = [
      FEATURE_TO_STRING[feature][int(encoded_val)] for encoded_val in row
  ]
  decoded_str = ''.join(decoded_row)
  return decoded_str


def to_acgt(encoded_array: np.ndarray, remove_gaps: bool = False) -> str:
  """Convert an encoded 1d array to a nucleotide string."""
  seq = np.array(encoded_array, dtype=np.int32)
  seq = ''.join(vocab_lookup(seq))
  if remove_gaps:
    seq = seq.replace('*', '')
  return seq


def get_sequences(
    batch: dict[str, Any],
    example_i: int,
    y_pred: Optional[np.ndarray] = None,
    remove_gaps: bool = False,
) -> dict[str, Union[str, list[str]]]:
  """Get easy ACGT sequences for ref, labels, and predictions."""
  features = break_example_into_feature_rows(batch['example'][example_i])
  sequences = {}
  sequences['reference'] = to_acgt(
      features['reference'][0], remove_gaps=remove_gaps
  )
  if 'label' in batch:
    labels = batch['label'][example_i]
    if len(labels.shape) == 2:
      sequences['label'] = [
          to_acgt(seq, remove_gaps=remove_gaps) for seq in labels
      ]
    else:
      sequences['label'] = to_acgt(labels, remove_gaps=remove_gaps)
  if y_pred is not None:
    y_pred_bases = np.argmax(y_pred, axis=-1)
    predictions = y_pred_bases[example_i]
    if len(predictions.shape) == 2:
      sequences['prediction'] = [
          to_acgt(seq, remove_gaps=remove_gaps) for seq in predictions
      ]
    else:
      sequences['prediction'] = to_acgt(predictions, remove_gaps=remove_gaps)

  return sequences
