#!/bin/bash

set -x

./build.sh
/usr/bin/cmake --build ./cmake-build-release --target demo_llm_run -- -j 6

run_file=./cmake-build-release/src/demo_llm_run

${run_file} 