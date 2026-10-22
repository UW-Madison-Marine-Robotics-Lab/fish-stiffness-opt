#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${FISH_SIM_BUILD_DIR:-${project_dir}/build}"
build_type="${FISH_SIM_BUILD_TYPE:-Release}"
python_command="${FISH_SIM_PYTHON:-python3}"
python_executable="$("${python_command}" -c 'import sys; print(sys.executable)')"

cmake -S "${project_dir}" -B "${build_dir}" \
    "-DCMAKE_BUILD_TYPE=${build_type}" \
    "-DPython_EXECUTABLE=${python_executable}" \
    "$@"
cmake --build "${build_dir}" --config "${build_type}" \
    --parallel "${FISH_SIM_BUILD_JOBS:-2}"
