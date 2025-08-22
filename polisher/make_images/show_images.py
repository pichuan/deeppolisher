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
"""Visualize images created with polisher make_images.

This script helps to visualize polisher images.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import colorama
import tensorflow.compat.v2 as tf

from polisher.make_images import encoding
from polisher.models import data_providers


_INPUT_IMAGES = flags.DEFINE_string(
    'input',
    None,
    'Path to input images to visualize',
)

_OUTPUT = flags.DEFINE_string(
    'output',
    None,
    'Path to output directory.',
)

_NUM_RECORDS = flags.DEFINE_integer(
    'num_records',
    -1,
    'Number of records to visualize. -1 means visualize all. Default is -1.',
)


def colorful(seq: str) -> str:
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


def show_example(example: tf.Tensor) -> None:
  """Show an example with example, hp_tag and labels."""
  # Prepare output directory:
  output_prefix = (
      '{}_'.format(_OUTPUT.value) if _OUTPUT.value is not None else ''
  )
  if output_prefix:
    tf.io.gfile.makedirs(os.path.dirname(output_prefix))
  # Output filename
  name = example['name'][0].numpy()[0].decode('utf-8')
  filename_txt = f'{output_prefix}{name}.txt'
  txt_output = tf.io.gfile.GFile(filename_txt, 'w')

  # Decode image and output
  feature_indices = data_providers.get_feature_indices()
  vocab = encoding.get_vocab()
  base_decoder = {encoding.get_nucleotide_encoding(v): v for v in vocab}
  feature_decoders = {
      'reference': base_decoder,
      'encoded_bases': base_decoder,
      'encoded_match_mismatch': {0: '-', 1: '.', 2: 'X'},
  }
  label = example['label'][0].numpy()
  decoded_label = [
      feature_decoders['encoded_bases'][int(encoded_val)]
      for encoded_val in label
  ]
  if not decoded_label:
    shape = example['example'][0].numpy().shape
    decoded_label = ['-'] * shape[1]
    decoded_str = ''.join(decoded_label)
  else:
    decoded_str = ''.join(decoded_label)
  txt_output.write('Label\t' + colorful(decoded_str) + '\n')

  for feature in feature_decoders:
    row_slice = feature_indices[feature]
    example_processed = example['example'][0]
    rows = example_processed[row_slice[0] : row_slice[1], :, :].numpy()
    for read_i, row in enumerate(rows):
      decoded_row = [
          feature_decoders[feature][int(encoded_val)] for encoded_val in row
      ]
      decoded_str = ''.join(decoded_row)
      output_str = ''
      if feature == 'encoded_bases':
        output_str = f'Read {read_i + 1}\t' + colorful(decoded_str) + '\n'
      elif feature == 'reference':
        output_str = 'Ref\t' + colorful(decoded_str) + '\n'
      elif feature == 'encoded_match_mismatch':
        output_str = f'M/X {read_i + 1}\t' + colorful(decoded_str) + '\n'
      txt_output.write(output_str)

  intensity_based_features = [
      'encoded_base_qualities',
      'encoded_mapping_quality',
  ]
  for feature in intensity_based_features:
    row_slice = feature_indices[feature]
    example_processed = example['example'][0]
    rows = example_processed[row_slice[0] : row_slice[1], :, :].numpy()
    for read_i, row in enumerate(rows):
      decoded_row = [str(int(encoded_val[0] / 10)) for encoded_val in row]
      decoded_str = ''.join(decoded_row)
      output_str = ''
      if feature == 'encoded_base_qualities':
        output_str = f'BQ {read_i + 1}\t' + decoded_str + '\n'
      elif feature == 'encoded_mapping_quality':
        output_str = f'MQ {read_i + 1}\t' + decoded_str + '\n'
      txt_output.write(output_str)


def visualize_images():
  """Visualize images."""
  dataset = data_providers.get_dataset(
      file_pattern=_INPUT_IMAGES.value,
      num_epochs=1,
      batch_size=1,
      limit=_NUM_RECORDS.value,
      drop_remainder=False,
      inference=False,
      example_label_tuple=False,
      shuffle_dataset=False,
  )
  for example in dataset:
    show_example(example)


def register_required_flags():
  flags.mark_flags_as_required(['images', 'output'])


def main(argv: Sequence[str]) -> None:
  del argv
  visualize_images()


if __name__ == '__main__':
  app.run(main)
