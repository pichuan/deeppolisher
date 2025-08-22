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
"""Classic non-AI assembler for inference.

This is a placeholder for the classic assembler, which is used for inference.
Add implementation of assemble_haplotypes_fn.
"""

from typing import Any, List
from absl import logging
import numpy as np


# NOLINTBEGIN
def assemble_haplotypes_fn(
    reads_data: List[str],
    reference_seq: str,
    base_quality_scores: List[
        np.ndarray
    ],  # Each ndarray contains quality scores for bases in one read
    read_mapping_quality_scores: List[
        np.ndarray
    ],  # Each ndarray contains a single mapping quality for one read
    read_haplotypes: List[
        np.ndarray
    ],  # Each ndarray contains a single haplotype tag for one read (0, 1, or 2)
    config: dict[str, Any],
) -> List[str]:
  """Assembles two haplotype sequences from reads data.

  This function is a placeholder for a classic (non-AI) assembler. It takes
  reads data, quality scores, and haplotype information to construct two
  haplotype sequences.

  Args:
    reads_data: A list of strings, where each string is a DNA sequence of a
      read.
    reference_seq: The reference DNA sequence as a string.
    base_quality_scores: A list of numpy arrays, where each array contains the
      base quality scores for a single read.
    read_mapping_quality_scores: A list of numpy arrays, where each array
      contains the mapping quality score for a single read.
    read_haplotypes: A list of numpy arrays, where each array contains the
      haplotype tag (0, 1, or 2) for a single read.
    config: A dictionary of configuration parameters for the assembler.

  Returns:
    A list containing two strings, representing the assembled haplotype
    sequences.
  """
  logging.info(
      reads_data,
      reference_seq,
      base_quality_scores,
      read_mapping_quality_scores,
      read_haplotypes,
      config,
  )
  raise NotImplementedError('assemble_haplotypes_fn is not implemented.')


# NOLINTEND
