# Fish Stiffness Optimization

Simulation and reinforcement learning for distributed passive stiffness optimization
in robotic fish.

## Citation

This repository accompanies the paper:

**[Dynamics Modeling and Swimming Optimization of Robotic Fish with Various Local Body Stiffness](https://ieeexplore.ieee.org/document/11753345)**  
Jiayi Jin, Li-Ming Chao, Liang Li, and Wei Wang.  
*IEEE Robotics and Automation Letters*, Early Access, pp. 1–8, 2026.  
DOI: [10.1109/LRA.2026.3746908](https://doi.org/10.1109/LRA.2026.3746908).

If you use this framework, please cite the paper using [BibTeX](CITATION.bib)
or [CITATION.cff](CITATION.cff).

## Install on Ubuntu or macOS

We recommend a dedicated [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
with Python 3.10. The simulator requires a C++17 compiler, CMake 3.18 or newer,
and Python 3.10 or newer with development headers. Eigen and pybind11 sources
are included, so the default build does not download C++ dependencies or require
BLAS/LAPACK packages.

Create and activate the environment. Conda supplies Python and its development
headers, pip, CMake, and FFmpeg for animation export:

```sh
conda create -n fish-stiffness-opt --override-channels -c conda-forge python=3.10 pip "cmake>=3.18" ffmpeg
conda activate fish-stiffness-opt
```

From the repository directory, install the simulation dependencies and build:

```sh
python -m pip install -r requirements.txt
./build_fish_sim.sh
python -c 'import fish_sim; print(fish_sim.__doc__)'
```

Activate this environment whenever you build or run the code. Install the Conda
packages first, then install the Python requirements with the environment's pip.
The requirement files pin direct dependency versions used in local Python 3.10
tests; pip resolves their transitive dependencies.

The build script reuses `build/` and places the compiled `fish_sim` module beside
the Python scripts. It can also be called from another working directory.
`FISH_SIM_PYTHON` selects a Python executable; `FISH_SIM_BUILD_DIR`,
`FISH_SIM_BUILD_TYPE`, and `FISH_SIM_BUILD_JOBS` override the build directory,
configuration, and number of parallel build jobs. For example:

```sh
FISH_SIM_PYTHON=/path/to/environment/bin/python ./build_fish_sim.sh
```

To use an installed BLAS library, enable it explicitly. On Ubuntu:

```sh
sudo apt install libopenblas-dev
./build_fish_sim.sh -DFISH_SIM_USE_BLAS=ON
```

CMake discovers the BLAS implementation, including Accelerate on macOS.
LAPACKE is not required. To return to the default backend, rebuild with
`./build_fish_sim.sh -DFISH_SIM_USE_BLAS=OFF`.

## Run a simulation

```sh
python simulate_fish.py
```

The example prints swimming speed and cost of transport and saves an animation
under `res/`. The Conda setup above includes FFmpeg for video export.

## Train and evaluate

The trainer uses the CPU. With the Conda environment active, Ubuntu users can
first install the
[CPU-only PyTorch wheel](https://pytorch.org/get-started/previous-versions/#v260)
to avoid downloading CUDA dependencies:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Then, on Ubuntu or macOS, install the RL dependencies:

```sh
python -m pip install -r requirements-rl.txt
```

This includes the simulation requirements plus PyTorch, Gymnasium,
Stable-Baselines3, and TensorBoard.

Start training:

```sh
python stiffness_rl.py
```

The default training run saves `res/ppo_fish_2.0.zip`. Evaluate that checkpoint
and export its swimming animation with:

```sh
python test_model.py --freq 2.0
```

Evaluation uses the training environment's speed objective and simulation rate.
It creates `res/video/` and saves `fish_2.0.mp4` there. Use `--model_dir` and
`--video_dir` to select different checkpoint and video directories. Train a model
before evaluation; pretrained checkpoints are not included.

## License

The project's original code is licensed under the [MIT License](LICENSE).

Bundled [Eigen](src/Eigen/COPYING.README) and
[pybind11](extern/pybind11/LICENSE) retain their upstream licenses.
