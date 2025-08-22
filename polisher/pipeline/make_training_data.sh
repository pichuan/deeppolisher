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

source gbash.sh || exit

set -euo pipefail


DEFINE_string --required make_images_dir "" "Directory name where images will be written."
DEFINE_string --required bam_gcs "" "GCS URI for BAM file."
DEFINE_string --required fasta_gcs "" "GCS URI for Fasta file."
DEFINE_string --required region_bed_gcs "" "GCS URI for Bed file."
DEFINE_string --required truth_to_ref_gcs "" "GCS URI for Truth BAM file."


DEFINE_string docker_image_tag "deeploid_experiment_$(date '+%Y%m%d')" "Tag to use for the docker image."
DEFINE_bool build_docker_image false  "If true, builds new docker image."
DEFINE_bool gcp_use_existing_image false  "If true, tries to start up an existing GCP instance with the same name."
DEFINE_string gcp_instance_name "deeploid-make-images" "GCP instance name"
DEFINE_string vm_zone "us-west1-b" "Zone where VM should be launched."



function create_docker_image() {
  tag=${FLAGS_docker_image_tag}
  ./learning/genomics/polisher/google/make_docker_image.sh --tag "${tag}"
  # Get polisher version to verify the docker image built correctly
  polisher_version=$(docker run -it -v /data:/data \
    gcr.io/google.com/brain-genomics/polisher:"${tag}" \
    polisher --version)
  echo "Polisher version: ${polisher_version}"
}


function run_make_examples_on_gcp() {
  VM=${FLAGS_gcp_instance_name}
  gcloud config set project google.com:brain-genomics
  if (( ${FLAGS_gcp_use_existing_image} )); then
    # try to start the existing GCP image
    gcloud compute instances start \
      "${VM}" \
      --zone "${FLAGS_vm_zone}"
  else 
    # create a new instance.
    gcloud compute instances create \
      "${VM}" \
      --scopes "compute-rw,storage-full,cloud-platform" \
      --image-family "ubuntu-2004-lts" \
      --image-project "ubuntu-os-cloud" \
      --zone "${FLAGS_vm_zone}" \
      --machine-type "n2-standard-128" \
      --boot-disk-size "1000" \
      --boot-disk-type "pd-ssd"
  fi

  # Check that $host can accept SSH connections
  # Idea came from: https://stackoverflow.com/a/29455821
  while ! gcloud compute ssh --project "google.com:brain-genomics" --zone="${FLAGS_vm_zone}" "${VM}" -- -o ConnectTimeout=5 -o StrictHostKeyChecking=no "echo 'up'" > /dev/null 2>&1
  do
    echo "Waiting for machine [$VM] to start. Sleep for 2 seconds."
    sleep 2
  done

  # Copy over the script
  gcp_script="learning/genomics/polisher/pipeline/gcp_make_training_data.sh"
  gcloud compute scp \
    --zone="${FLAGS_vm_zone}" \
    $gcp_script "${VM}":.
  gcloud compute ssh "${VM}" \
    --command "chmod +x gcp_make_training_data.sh" \
    --zone="${FLAGS_vm_zone}"

  gcp_command="./gcp_make_training_data.sh \
    --docker_image_tag ${FLAGS_docker_image_tag} \
    --make_images_dir ${FLAGS_make_images_dir} \
    --bam_gcs ${FLAGS_bam_gcs} \
    --fasta_gcs ${FLAGS_fasta_gcs} \
    --region_bed_gcs ${FLAGS_region_bed_gcs} \
    --truth_to_ref_gcs ${FLAGS_truth_to_ref_gcs}"
  
  # Run Make examples
  gcloud compute ssh "${VM}" \
    --command "${gcp_command}" \
    --zone="${FLAGS_vm_zone}"
  
  # Stop the instance
  gcloud compute instances stop "${VM}" --zone "${FLAGS_vm_zone}"
}

function shuffle_examples() {
  echo "Shuffling examples"
  INPUT_PATH=/bigstore/brain-genomics/${USER}/deeploid/${FLAGS_make_images_dir}/make_images_outputs
  DATA_PATH="home/brain-genomics/deepconsensus/gs/brain-genomics/${USER}/deeploid/${FLAGS_make_images_dir}/make_images_outputs"
  OUTPUT_PATH=<internal>

  time blaze run -c opt \
    //learning/genomics/deepconsensus/pipelines/tasks:shuffle.par -- \
      --input_path "${INPUT_PATH}" \
      --output_path "${OUTPUT_PATH}" \
      --flume_exec_mode=BORG \
      --flume_borg_user_name="${USER}" \
      --flume_borg_accounting_charged_user_name=brain-genomics \
      --flume_batch_scheduler_strategy=RUN_SOON \
      --flume_use_batch_scheduler \
      --flume_worker_priority=100 \
      --flume_close_to_resources="<internal>
      --flume_backend=DAX \
      --flume_auto_retry=false \
      --flume_tmp_file_cells="is-d" \
      --flume_tmp_dir_group="brain-genomics" \
      --flume_completion_email_address="${USER}@google.com" \
      --bigstore_robot_account="${USER}@system.gserviceaccount.com"

  # Propagate shuffled examples to placer
  placer publish_existing \
    <internal>
    <internal>
}


function main() {
  # Create Docker image
  if (( ${FLAGS_build_docker_image} )); then
    create_docker_image
  fi
  # Run make_examples on GCP
  run_make_examples_on_gcp
  # Shuffle examples
  shuffle_examples
}


gbash::main "$@"
