# moka-rlbench

> **An unofficial RLBench port of MOKA** ([Liu et al., RSS 2024](https://arxiv.org/abs/2403.03174)).
> Originally hardware-only and ROS-dependent, MOKA is reimplemented here as a fully simulated, reproducible baseline for VLM-driven robotic manipulation.

<p align="center">
  <a href="https://arxiv.org/abs/2403.03174"><img alt="Paper" src="https://img.shields.io/badge/paper-RSS%202024-red"></a>
  <a href="https://moka-manipulation.github.io"><img alt="Website" src="https://img.shields.io/badge/website-moka-blue"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10-blue"></a>
  <a href="#"><img alt="RLBench" src="https://img.shields.io/badge/sim-RLBench-orange"></a>
  <a href="#"><img alt="VLM" src="https://img.shields.io/badge/VLM-GPT--4V-green"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

---

## Disclaimer

This is an **independent, unofficial reimplementation** of [MOKA](https://moka-manipulation.github.io/), ported from the authors' real-robot codebase to the [RLBench](https://github.com/stepjam/RLBench) physics simulator. It is **not** affiliated with or endorsed by the original authors. All credit for the MOKA framework and methodology belongs to the original team.

This port was developed (Oct 2025 – Dec 2025) as a baseline for our IROS 2026 submission **RADAR** ([arXiv:2603.11811](https://arxiv.org/abs/2603.11811)).

---

## What MOKA does

MOKA ([Liu et al., RSS 2024](https://arxiv.org/abs/2403.03174)) solves language-conditioned manipulation by asking a VLM to **point at pixels** rather than predict actions:

1. Render the scene from a camera (RGB + depth)
2. Overlay candidate keypoints as a **Set-of-Mark** visual prompt
3. Ask GPT-4V to pick the **grasp keypoint**, **function keypoint**, **target keypoint**, and a coarse motion plan from natural-language instructions
4. Lift the chosen 2D keypoints into 3D and execute a scripted motion primitive

This port preserves the original 2D-keypoint-centric design, only swapping the perception/actuation stack for RLBench.

---

## Why this port?

The original MOKA release targets a physical Franka robot with [DROID](https://droid-dataset.github.io/) platform and ROS-specific dependencies, making head-to-head benchmarking against simulator-native baselines difficult. `moka-rlbench` removes that barrier by providing:

- A **fully simulated, ROS-free** MOKA pipeline that runs end-to-end inside RLBench
- A **VLM ↔ simulator bridge** wiring RLBench camera renders directly into the GPT-4V API
- A **deterministic, reproducible evaluation harness** for multi-step manipulation tasks
- **Per-step intermediate artifacts** (Set-of-Mark images, keypoint plans, motion overlays) for transparent debugging and qualitative analysis

If you are comparing a new VLM-based manipulation method against MOKA in simulation, this repo is meant to be a drop-in baseline.

---

## Pipeline overview

```
┌────────────────┐   ┌─────────────────┐   ┌───────────────────┐   ┌──────────────┐
│  RLBench task  │ → │ Set-of-Mark     │ → │  GPT-4V planner   │ → │ keypoint →   │
│  (RGB + depth) │   │ candidate marks │   │ (grasp / target / │   │ 3D motion in │
│                │   │ on obs image    │   │  motion plan)     │   │ RLBench      │
└────────────────┘   └─────────────────┘   └───────────────────┘   └──────────────┘
```

Per task step the runner emits:

| File | Purpose |
|------|---------|
| `obs_image.jpg` | Raw camera render from RLBench |
| `annotated_image.jpg` | Same view with Set-of-Mark candidate keypoints |
| `plan.json` | High-level VLM plan (instruction, grasped object, motion direction) |
| `context.json` | Selected keypoints, target tile, pre/post-contact tiles, heights, angle |
| `motion_image.jpg` | Motion-plan overlay used for qualitative inspection |

---

## Installation

> Tested on Ubuntu 20.04 with CUDA 11.8, Python 3.10, and PyTorch 2.1.1.

The full stack has four pieces. Install them in this order:

1. **CoppeliaSim 4.1** (required by PyRep)
2. **PyRep + RLBench** (built from upstream source)
3. **PyTorch matching your CUDA** (separate install so you pick the right wheel)
4. **Everything else** (`requirements.txt`)

### 1. Clone and create the environment

```bash
git clone https://github.com/kite-sense/moka_rlbench.git
cd moka_rlbench

conda create -n moka python=3.10
conda activate moka
pip install -r requirements.txt
pip install -e .
```

### 2. Install CoppeliaSim 4.1, PyRep, and RLBench

These cannot be `pip install`-ed directly. Follow the upstream instructions:

- CoppeliaSim 4.1: https://www.coppeliarobotics.com/previousVersions
- PyRep: https://github.com/stepjam/PyRep
- RLBench: https://github.com/stepjam/RLBench

Quick reference (Ubuntu, after CoppeliaSim is extracted to `~/CoppeliaSim`):

```bash
export COPPELIASIM_ROOT=~/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

git clone https://github.com/stepjam/PyRep.git && cd PyRep
pip install -e .
cd ..

git clone https://github.com/stepjam/RLBench.git && cd RLBench
pip install -e .
cd ..
```

### 3. Install PyTorch matching your CUDA

Pick the command for your platform from https://pytorch.org/get-started/locally/.
The reference environment used CUDA 11.8:

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Grounded-SAM and dependencies

The Grounded-SAM submodule is included in this repository under `Grounded-Segment-Anything/`.

```bash
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip uninstall roboflow supervision
pip install roboflow supervision==0.15.0
cd ..
```

### 5. Install detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 6. Download checkpoints

Download the DINO and SAM checkpoints into the `ckpts/` directory:

```bash
mkdir -p ckpts && cd ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

### 7. Configure VLM access

```bash
export OPENAI_API_KEY="sk-..."
```

### Reproducibility tiers

| File | When to use |
|------|-------------|
| `environment.yml` + `requirements.txt` | **Recommended.** Loose version bounds, works across different CUDA versions. |
| `requirements-lock.txt` | **Paper-level reproduction.** Pinned to the exact wheels used in our experiments (Ubuntu 20.04, CUDA 11.8, Python 3.10). |
| `requirements-dev.txt` | **Development only.** Adds Jupyter, black, yapf on top of `requirements.txt`. |

---

## Quick start

Check `demo_phone_base.ipynb` for a quick start on visual prompting in the RLBench environment.

Run the full MOKA pipeline on RLBench tasks:

```bash
python run_rlbench.py --config config/rlbench_config.yaml
```

For evaluation with the Franka robot (original MOKA setup):

```bash
python franka_eval.py --config config/moka.yaml --data_dir /path/to/dataset
```

---

## Benchmark and results

The port is benchmarked on **10 RLBench tasks** — 7 atomic and 3 long-horizon — chosen to span single-object pick-and-place, articulated manipulation, and multi-step skill chaining. This task set is identical to the one used in the RADAR paper ([arXiv:2603.11811](https://arxiv.org/abs/2603.11811)), so numbers from this port are directly comparable to the published baselines.

### Evaluation protocol

- **Simulator:** RLBench
- **Rollouts:** 10 per task, with randomized initial states between rollouts
- **Reset:** simulation ground truth between rollouts (so the protocol isolates *forward* generation and execution, not the reset mechanism)
- **VLM context:** 1-shot — a single high-quality demonstration provided as in-context prior. Increasing the number of demonstrations did not yield proportional gains.
- **VLM backbone:** GPT-4V via the OpenAI API. CLIP was tested for skill retrieval but discarded — its embeddings are heavily noun-biased and fail to distinguish fine-grained actionable semantics (e.g., "lift cup" vs. "translate cup").

### Tasks

#### Atomic (7)

| Task | Description |
|------|-------------|
| Large Container (Cup) | Place a cup into a large open container |
| Large Container (Block) | Place a block into a large open container |
| Large Container (Laptop) | Place a laptop into a large open container |
| Push Block | Push a block to a target region |
| Stack Block | Stack one block on top of another |
| Close Box | Close an articulated box lid |
| Open Box | Open an articulated box lid |

#### Long-horizon (3)

| Task | Description |
|------|-------------|
| Put Laptop & Cup into Tray | Sequentially place two distinct objects into a shared tray |
| Push & Stack Blocks | First push a block to a region, then stack a second block on it |
| Close then Open Box | State-dependent articulated chaining (must close before re-opening) |

Cached qualitative artifacts (`obs_image`, `annotated_image`, `motion_image`, `plan.json`, `context.json`) for additional exploratory tasks (`Basketball`, `block_in_box`, `phone_on_base`, `stack_block`, `trash_in_bin`) ship under `tasks/` as references for the per-step pipeline output format.

### Results

Success rates (10 rollouts per task, 1-shot context) reproduced through this port. RADAR numbers are reported alongside, taken from the RADAR paper for direct comparison.

#### Atomic

| Task | MOKA *(this port)* | RADAR |
|------|:------:|:------:|
| Large Container (Cup) | 0.20 | **0.80** |
| Large Container (Block) | 0.30 | **0.80** |
| Large Container (Laptop) | 0.10 | **0.90** |
| Push Block | 0.40 | **1.00** |
| Stack Block | 0.10 | **0.80** |
| Close Box | 0.30 | **1.00** |
| Open Box | 0.20 | **0.70** |

#### Long-horizon

| Task | MOKA *(this port)* | RADAR |
|------|:------:|:------:|
| Put Laptop & Cup into Tray | 0.00 | **0.80** |
| Push & Stack Blocks | 0.00 | **0.40** |
| Close then Open Box | 0.10 | **0.90** |

Numbers are reproduced from Table I of the RADAR paper. Each cell is the success rate over 10 independent rollouts.

For the full analysis of MOKA's failure modes (2D → 3D back-projection error, long-horizon collapse, articulated / contact-rich limits), the ablations, and the real-robot deployment that motivates these comparisons, see our RADAR paper: **[arXiv:2603.11811](https://arxiv.org/abs/2603.11811)**.

---

## Repository layout

```
moka_rlbench/
├── moka/                          # Core MOKA module
│   ├── drivers/                   # Robot/simulator drivers (driver.py, new_driver.py)
│   ├── planners/                  # Visual-prompt planner + utils
│   ├── policies/                  # Franka policy (action execution)
│   ├── vision/                    # Segmentation, keypoints, depth, grasp, visualization
│   ├── utils/                     # Config, time, transformation utilities
│   └── gpt_utils.py               # GPT-4V API helpers
├── Grounded-Segment-Anything/     # Grounded-SAM submodule (perception backbone)
├── config/                        # YAML configs (moka.yaml, rlbench_config.yaml)
├── prompts/                       # System/user prompt templates for GPT-4V
├── example/                       # Example observation images
├── ckpts/                         # Model checkpoints (SAM, GroundingDINO)
├── demo_phone_base.ipynb          # Quick-start notebook for RLBench
├── demo.py                        # Standalone demo script
├── run_rlbench.py                 # RLBench entry point
├── franka_eval.py                 # Real-robot evaluation entry (original MOKA)
├── rlbench_env_wrapper.py         # RLBench environment wrapper
├── ip_tasks.py                    # IP-task definitions
├── requirements.txt               # Runtime dependencies
├── environment.yaml               # Conda environment spec
└── setup.py                       # Package install config
```

---

## Citation

If you use this port in academic work, please cite **both** the original MOKA paper and our RADAR follow-up:

```bibtex
@inproceedings{liu2024moka,
  title     = {MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting},
  author    = {Liu, Fangchen and Fang, Kuan and Abbeel, Pieter and Levine, Sergey},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2024}
}

@article{wang2026radar,
  title   = {RADAR: Robot Autonomous Data Acquisition and Resetting via Few-Shot Imitation},
  author  = {Wang, Yongzhong and Zhu, Keyu and Zhong, Yong and Wang, Liqiong and Yang, Jinyu and Zheng, Feng},
  journal = {arXiv preprint arXiv:2603.11811},
  year    = {2026}
}
```

If you reference *this port specifically*, please also link back to the repository:

```bibtex
@misc{wang2025mokarlbench,
  title        = {moka-rlbench: An unofficial RLBench port of MOKA},
  author       = {Wang, Yongzhong},
  year         = {2025},
  howpublished = {\url{https://github.com/kite-sense/moka_rlbench}}
}
```

---

## Acknowledgments

- The original [MOKA](https://moka-manipulation.github.io/) authors (Kuan Fang, Fangchen Liu, Pieter Abbeel, Sergey Levine) for the open-source release that made this port possible.
- The perception pipeline is adapted from [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [SAM](https://arxiv.org/abs/2304.02643).
- The [RLBench](https://github.com/stepjam/RLBench) and [PyRep](https://github.com/stepjam/PyRep) maintainers for the simulation infrastructure.
- Visual-prompting inspiration from [SoM](https://github.com/microsoft/SoM).
- VIP Lab, SUSTech, where the IROS 2026 RADAR project this port supports was developed.

---

## License

Released under the MIT License. See [LICENSE](LICENSE). Note that RLBench, PyRep, Grounded-SAM, and OpenAI's API each carry their own licenses and terms of use — comply with all of them when using this code.
