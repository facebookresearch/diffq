#!/bin/bash

cd "$(dirname "$0")"

patch=$(ls deit.*.patch)
commit=echo $path | cut -d. -f2

echo "Initialization of submodules"
git submodule update --init --recursive
cd deit
echo "Checking out $commit"
git checkout $commit
echo "Applying patch"
git apply ../$patch