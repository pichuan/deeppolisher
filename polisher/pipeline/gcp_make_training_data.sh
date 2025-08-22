#!/bin/bash
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

CPUS=127
PLOIDY=2
# Parse command line args in format "--option foo"
for i in "$@"; do
  case $i in
    --docker_image_tag)
      docker_image_tag="$2"
      shift
      shift
      ;;
    --make_images_dir)
      make_images_dir="$2"
      shift
      shift
      ;;
    --bam_gcs)
      bam_gcs="$2"
      bam=$(basename "$bam_gcs")
      shift
      shift
      ;;
    --fasta_gcs)
      fasta_gcs="$2"
      fasta=$(basename "$fasta_gcs")
      shift
      shift
      ;;
    --truth_to_ref_gcs)
      truth_to_ref_gcs="$2"
      truth_to_ref=$(basename "$truth_to_ref_gcs")
      shift
      shift
      ;;
    --region_bed_gcs)
      region_bed_gcs="$2"
      region_bed=$(basename "$region_bed_gcs")
      shift
      shift
      ;;
    *)
      ;;
  esac
done


function make_images_inference() {
  sudo docker run -v "${HOME}/${make_images_dir}":/data \
    gcr.io/google.com/brain-genomics/polisher:"${docker_image_tag}" \
    polisher make_images \
    --bam "/data/${bam}" \
    --fasta "/data/${fasta}" \
    --output "/data/inference_make_images_output" \
    --cpus "${CPUS}" \
    --region "chr20" \
    --ploidy "${PLOIDY}" \
    --logtostderr 2>&1 | tee "${HOME}/${make_images_dir}_make_images_inference.log"
}

function make_images_training() {
  sudo docker run -v "${HOME}/${make_images_dir}":/data \
    gcr.io/google.com/brain-genomics/polisher:"${docker_image_tag}" \
    polisher make_images \
    --bam "/data/${bam}" \
    --fasta "/data/${fasta}" \
    --output "/data/train_make_images_output" \
    --cpus "${CPUS}" \
    --training_mode \
    --truth_to_ref "/data/${truth_to_ref}" \
    --region "chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9" \
    --region_bed "/data/${region_bed}" \
    --ploidy "${PLOIDY}" \
    --logtostderr 2>&1 | tee "${HOME}/${make_images_dir}_make_images_training.log"
}

function make_images_tuning() {
  sudo docker run -v "${HOME}/${make_images_dir}":/data \
    gcr.io/google.com/brain-genomics/polisher:"${docker_image_tag}" \
    polisher make_images \
    --bam "/data/${bam}" \
    --fasta "/data/${fasta}" \
    --output "/data/tune_make_images_output" \
    --cpus "${CPUS}" \
    --training_mode \
    --truth_to_ref "/data/${truth_to_ref}" \
    --region "chr21,chr22" \
    --region_bed "/data/${region_bed}" \
    --ploidy "${PLOIDY}" \
    --logtostderr 2>&1 | tee "${HOME}/${make_images_dir}_make_images_tuning.log"
}

function copy_examples_to_gcs() {
  # Copy data over to GCS, keeping train/eval/inference in separate folders like the shuffle step expects.
  # The shuffle step also expects to find ./train and ./eval folders, so we are renaming tune to eval here.
  gcs_output_path=gs://brain-genomics/${USER}/deeploid/${make_images_dir}/make_images_outputs

  # Rename examples from *.tfrecords.gz to *.tfrecord.gz
  pattern="${HOME}/${make_images_dir}"/*_make_images_output_*.tfrecords.gz
  for f in $pattern; do
      mv -- "$f" "${f%.tfrecords.gz}.tfrecord.gz"
  done
  
  gsutil -m cp "${HOME}/${make_images_dir}"/inference* "${gcs_output_path}"/inference/
  gsutil -m cp "${HOME}/${make_images_dir}"/train* "${gcs_output_path}"/train/
  gsutil -m cp "${HOME}/${make_images_dir}"/tune* "${gcs_output_path}"/eval/

  # Write training summary
  n_examples_train=$(gsutil cat "${gcs_output_path}"/train/train_make_images_output_training.summary.json | grep "example_counter" | tr -d -c 0-9)
  n_examples_eval=$(gsutil cat "${gcs_output_path}"/eval/tune_make_images_output_training.summary.json | grep "example_counter" | tr -d -c 0-9)
  echo """
  {
    \"n_examples_train\": ${n_examples_train},
    \"n_examples_eval\": ${n_examples_eval}
  }
  """ > summary.training.json
  # Copy to GCS
  gsutil cp ~/summary.training.json "${gcs_output_path}"/summary.training.json
}


# Install docker
sudo apt -y update
sudo apt-get -y install docker.io

# Authenticate with gcr.io for access to BG internal docker images:
gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin https://gcr.io

# Pull the docker image you made:
sudo docker pull "gcr.io/google.com/brain-genomics/polisher:${docker_image_tag}"

# Copy all the input data onto the machine
cd
mkdir -p "${make_images_dir}"
sudo chmod -R o+rwx "${make_images_dir}"
gsutil -m cp "${bam_gcs}"{,.bai} "${make_images_dir}"
gsutil -m cp "${truth_to_ref_gcs}"{,.bai} "${make_images_dir}"
gsutil -m cp "${region_bed_gcs}" "${make_images_dir}"
gsutil -m cp "${fasta_gcs}"{,.fai} "${make_images_dir}"

# Run make_images for inference, training and tuning
make_images_inference
make_images_training
make_images_tuning

# Finally, copy examples to GCS
copy_examples_to_gcs
