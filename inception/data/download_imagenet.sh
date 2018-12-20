#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to download ImageNet Challenge 2012 training and validation data set.
#
# Downloads and decompresses raw images and bounding boxes.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  ./download_imagenet.sh [dir name] [synsets file]
set -e

echo "cc"
echo `pwd`

IMAGENET_ACCESS_KEY="741c0b920664186b343cf8ffa2291e1726a79aa2"
IMAGENET_USERNAME="chenc10"

if [ "x$IMAGENET_ACCESS_KEY" == x -o "x$IMAGENET_USERNAME" == x ]; then
  cat <<END
In order to download the imagenet data, you have to create an account with
image-net.org. This will get you a username and an access key. You can set the
IMAGENET_USERNAME and IMAGENET_ACCESS_KEY environment variables, or you can
enter the credentials here.
END
  read -p "Username: " IMAGENET_USERNAME
  read -p "Access key: " IMAGENET_ACCESS_KEY
fi

echo $1
OUTDIR="${1:-./imagenet-data}"
echo "start"
echo $OUTDIR
echo $2
#OUTDIR="/home/cchen/models-master/research/inception"
SYNSETS_FILE="${2:-./synsets.txt}"

echo "c1: "`pwd`
echo "Saving downloaded files to $OUTDIR"
mkdir -p "${OUTDIR}"
CURRENT_DIR=$(pwd)
BBOX_DIR="${OUTDIR}bounding_boxes"
mkdir -p "${BBOX_DIR}"
#cd "${OUTDIR}"

# Download and process all of the ImageNet bounding boxes.
BASE_URL="http://www.image-net.org/challenges/LSVRC/2012/nonpub"

# See here for details: http://www.image-net.org/download-bboxes
BOUNDING_BOX_ANNOTATIONS="${BASE_URL}/ILSVRC2012_bbox_train_v2.tar.gz"
BBOX_TAR_BALL="${BBOX_DIR}/annotations.tar.gz"
echo "Downloading bounding box annotations."
wget "${BOUNDING_BOX_ANNOTATIONS}" -O "${BBOX_TAR_BALL}" || BASE_URL_CHANGE=1
if [ $BASE_URL_CHANGE ]; then
  BASE_URL="http://www.image-net.org/challenges/LSVRC/2012/nnoupb"
  BOUNDING_BOX_ANNOTATIONS="${BASE_URL}/ILSVRC2012_bbox_train_v2.tar.gz"
  BBOX_TAR_BALL="${BBOX_DIR}/annotations.tar.gz"
fi
wget "${BOUNDING_BOX_ANNOTATIONS}" -O "${BBOX_TAR_BALL}"
echo "Uncompressing bounding box annotations ..."
echo `pwd`
tar xzf "${BBOX_TAR_BALL}" -C "${BBOX_DIR}"

LABELS_ANNOTATED="${BBOX_DIR}/*"
NUM_XML=$(ls -1 ${LABELS_ANNOTATED} | wc -l)
echo "Identified ${NUM_XML} bounding box annotations."

# Download and uncompress all images from the ImageNet 2012 validation dataset.
VALIDATION_TARBALL="ILSVRC2012_img_val.tar"
OUTPUT_PATH="${OUTDIR}validation/"
echo "OUTPUT_PATH: "$OUTPUT_PATH
mkdir -p "${OUTPUT_PATH}"
echo "c2: "`pwd`
#cd "${OUTDIR}/.."
echo "c3: "`pwd`
echo "Downloading ${VALIDATION_TARBALL} to ${OUTPUT_PATH}."
#wget -nd -c "${BASE_URL}/${VALIDATION_TARBALL}"
tar xf "${OUTDIR}../${VALIDATION_TARBALL}" -C "${OUTPUT_PATH}"

# Download all images from the ImageNet 2012 train dataset.
TRAIN_TARBALL="${OUTDIR}../ILSVRC2012_img_train.tar"
OUTPUT_PATH="${OUTDIR}train/"
mkdir -p "${OUTPUT_PATH}"
#cd "${OUTDIR}/.."
echo "Downloading ${TRAIN_TARBALL} to ${OUTPUT_PATH}."
#wget -nd -c "${BASE_URL}/${TRAIN_TARBALL}"

# Un-compress the individual tar-files within the train tar-file.
echo "Uncompressing individual train tar-balls in the training data."
TRAIN_TARBALL="../ILSVRC2012_img_train.tar"
SYNSETS_FILE="`pwd`/${SYNSETS_FILE}"
echo `pwd`
echo "file: ${SYNSETS_FILE}"
cd ${OUTDIR}
OUTPUT_PATH="train/"
while read SYNSET; do
  echo "Processing: ${SYNSET}"

  # Create a directory and delete anything there.
  mkdir -p "${OUTPUT_PATH}${SYNSET}"
  rm -rf "${OUTPUT_PATH}${SYNSET}/*"

  # Uncompress into the directory.
  echo "uncompress: ${TRAIN_TARBALL} to ${SYNSET}.tar"
  tar xf "${TRAIN_TARBALL}" "${SYNSET}.tar"
  tar xf "${SYNSET}.tar" -C "${OUTPUT_PATH}${SYNSET}/"
  rm -f "${SYNSET}.tar"

  echo "Finished processing: ${SYNSET}"
done < "${SYNSETS_FILE}"