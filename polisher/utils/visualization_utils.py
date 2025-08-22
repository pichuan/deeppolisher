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
"""Utilities for Colab visualization of Deeploid/Deepolisher.

Example usage in this notebook: ../colab_utils_notebook.ipynb
"""

from collections.abc import Callable
from typing import Any, Optional, Union

import colorama
import numpy as np

from polisher.utils import colab_utils
from google3.third_party.nucleus.util import vis


def colorful(seq: Union[str, list[str]]) -> str:
  """Add colors to a sequence of DNA."""
  fore = colorama.Fore
  background = colorama.Back
  colors = {
      'A': fore.GREEN,
      'C': fore.BLUE,
      'G': fore.YELLOW,
      'T': fore.RED,
      'X': fore.RED,
  }
  reset = fore.BLACK + background.RESET
  colored_seq = [f'{colors.get(base, reset)}{base}{reset}' for base in seq]
  return ''.join(colored_seq)


def diffs_colorful(ref: str) -> Callable[[Any], Any]:
  """Add colors to a sequence of DNA if bases don't match the reference."""

  def colorful_if_not_ref(decoded_str):
    fore = colorama.Fore
    background = colorama.Back
    colors = {
        'A': fore.GREEN,
        'C': fore.BLUE,
        'G': fore.YELLOW,
        'T': fore.RED,
        'X': fore.RED,
    }
    boring = fore.WHITE + background.RESET
    reset = fore.BLACK + background.RESET
    colored_seq = []
    for base, ref_base in zip(decoded_str, ref):
      if base != ref_base:
        colored_seq.append(f'{colors.get(base, reset)}{base}{reset}')
      else:
        colored_seq.append(f'{boring}{base}{reset}')
    return ''.join(colored_seq)

  return colorful_if_not_ref


def visualize_features(
    features: dict[str, Any],
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    string_mode: bool = True,
    only_acgt: bool = False,
    highlights: bool = False,
):
  """Show a visualization based on the given info about an example."""
  ref = colab_utils.decode_to_string(features['reference'][0, :], 'reference')
  color_function = diffs_colorful(ref) if highlights else colorful

  if predictions is not None:
    if len(predictions.shape) == 1:
      # pad haploid:
      predictions = np.expand_dims(predictions, axis=0)
    if string_mode:
      for i in range(predictions.shape[0]):
        decoded_prediction = [
            colab_utils.FEATURE_TO_STRING['encoded_bases'][int(encoded_val)]
            for encoded_val in predictions[i, :]
        ]
        print(f'Pred {i + 1}\t{color_function(decoded_prediction)}')
    else:
      print('Predictions:')
      vis.array_to_png(predictions)
  if labels is not None:
    if len(labels.shape) == 1:
      # pad haploid:
      labels = np.expand_dims(labels, axis=0)
    if string_mode:
      for i in range(labels.shape[0]):
        decoded_label = [
            colab_utils.FEATURE_TO_STRING['encoded_bases'][int(encoded_val)]
            for encoded_val in labels[i, :]
        ]
        print(f'Label {i + 1}\t{color_function(decoded_label)}')
    else:
      print('Labels:')
      vis.array_to_png(labels)

  for feature, rows in features.items():
    # Show each feature as string if decoder exists, otherwise as a heatmap.
    if only_acgt and feature not in ['reference', 'encoded_bases']:
      continue
    if string_mode and feature in colab_utils.FEATURE_TO_STRING:
      for read_i, row in enumerate(rows):
        decoded_str = colab_utils.decode_to_string(row, feature)
        if feature == 'encoded_bases':
          print(f'Read {read_i + 1}\t{color_function(decoded_str)}')
        elif feature == 'reference':
          print(f'Ref\t{colorful(decoded_str)}')
        elif feature == 'encoded_match_mismatch':
          print(f'M/X {read_i + 1}\t{colorful(decoded_str)}')
        # elif feature == 'encoded_hp_tag':
        #   print(f'HP {read_i + 1}\t', colorful(decoded_str))
    else:
      print(feature)
      # Set vmin and vmax to actual min/max possible for the data type to
      # standardize color scale between examples.
      vis.array_to_png(rows)  # vmin=0, vmax=100


def show_example(
    batch: dict[str, Any],
    example_i: int,
    y_pred: Optional[np.ndarray] = None,
    string_mode: bool = True,
    only_acgt: bool = True,
    highlights: bool = True,
) -> None:
  """Show an example in an easily human-readable way."""
  print(batch['name'][example_i].numpy()[0].decode('utf-8'))
  if 'labels' in batch:
    labels = batch['labels'][example_i]
  elif 'label' in batch:
    labels = batch['label'][example_i]
  else:
    labels = None
  features = colab_utils.break_example_into_feature_rows(
      batch['example'][example_i]
  )
  if y_pred is not None:
    y_pred_bases = np.argmax(y_pred, axis=-1)
    predictions = y_pred_bases[example_i]
  else:
    predictions = None
  visualize_features(
      features,
      labels=labels,
      predictions=predictions,
      string_mode=string_mode,
      only_acgt=only_acgt,
      highlights=highlights,
  )
