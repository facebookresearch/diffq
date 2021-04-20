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

cd $repo

if [[ -n $(git clean -n) ]]; then
    echo "Repo is not clean, commit recent changes"
    exit 1
fi
git diff $commit > ../$repo.$commit.patch

