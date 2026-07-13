#!/bin/bash
# Build mitsuba3 (sample-matching branch) with GPU + LLVM + scalar variants.
# Requirements: gcc 13.x, cmake 3.26.x (NOT cmake >= 4), ninja, CUDA >= 12,
# and the conda env from mi39-dev.yml activated.
set -e
cd "$(dirname "$0")/../.."   # expects mitsuba3/ checked out next to this repo
cd mitsuba3
git submodule update --init --recursive
mkdir -p build && cd build
cmake -GNinja \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DMI_DEFAULT_VARIANTS="scalar_rgb;llvm_ad_rgb;cuda_ad_rgb" ..
ninja
echo "Done. Activate with: source $(pwd)/setpath.sh"
# CPU backend at runtime: export DRJIT_LIBLLVM_PATH=<path to libLLVM .so>
