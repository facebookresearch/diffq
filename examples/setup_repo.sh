#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cd "$(dirname "$0")"
repo=$1

patch=$(ls $1.*.patch)
commit=$(echo $patch | cut -d. -f2)

./reset_repo.sh $repo || exit 1
cd $repo
echo "Applying patch"
git apply ../$patch
git add .
git commit -m "DiffQ support"
