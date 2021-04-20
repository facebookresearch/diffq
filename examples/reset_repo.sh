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

echo "Initialization of submodules"
git submodule update --init --recursive ./$repo || exit 1
cd $repo
echo "Checking if repo is clean"
if [[ -n $(git clean -n) ]]; then
    echo "Repo is not clean, please review files with 'git clean -n' and remove them with 'git clean -f'"
    exit 1
fi
echo "Checking out $commit"
git checkout -f $commit || exit 1
