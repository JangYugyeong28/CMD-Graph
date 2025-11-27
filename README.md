# Generating Command Modeling and Design Graphs with Data Augmentation for Enhanced 3D Modeling Support

<img width="595" height="250" alt="Figure1" src="https://github.com/user-attachments/assets/87d7da6a-f9e2-4849-bb5f-df9caca59d6e" />

[[ScienceDirect]](https://www.sciencedirect.com/science/article/abs/pii/S1474034625005373)  
**Authors:** [Yugyeong Jang](https://yugyeong.cargo.site/) · [Kyung Hoon Hyun](https://designinformatics.hanyang.ac.kr/People_Kyung-Hoon-Hyun)

---

## 🔍 Overview

This repository provides the **inference and post-processing code for extracting modeling command sequences** from 3D shapes.

Built on top of the original **shape2prog** pipeline, this code:

- decodes voxel shapes into a **domain-specific language (DSL)** program,
- converts the DSL into **Rhino-style modeling commands**, and
- optionally renders **reconstructed meshes and per-step sequence images**.

> 📝 This repository focuses on **modeling sequence extraction only**.  
> It does **not** include training code or dataset distribution, and it does **not** build CMD-Graph structures.

---

## 🗂 What This Repository Provides

This repo includes:

- ✅ **Modified test script (`test_copy.py`)**
  - Loads a trained shape2prog model
  - Decodes voxel shapes into programs (DSL)
  - Optionally saves:
    - DSL text files (`programs/`)
    - Converted Rhino command files (`rhino_commands/`)
    - Reconstructed model images (`images/`)
    - Step-by-step sequence images (`sequence_images/`)

- ✅ **Example shell command** for running creative design / inference

This repo **does not** provide:

- ❌ Training code for the program generator  
- ❌ The original datasets used in training  
- ❌ CMD-Graph construction code

Instead, it focuses on:

> **“Given 3D shapes (from a trained shape2prog model), how do we extract and save the corresponding modeling command sequences?”**

---

## 📦 Depends On: shape2prog

We build directly on:

- **shape2prog (Huang et al.)**  
  🔗 https://github.com/HobbitLong/shape2prog

In our setup:

- We follow the **same dataset configuration and training pipeline** as shape2prog.
- Voxel shapes are upsampled to **64×64×64** for training and decoding.
- The trained model checkpoint is then used by `test_copy.py` to generate programs.

Because of licensing and size constraints, we do **not** re-distribute:

- the original datasets  
- the full training code / checkpoints

Please follow the original shape2prog repository for:

- dataset preparation  
- training instructions  
- baseline testing scripts

---

## 🚀 Usage Guide — Modeling Sequence Extraction

Once you have shape2prog installed, trained, and your checkpoint ready,  
you can run the **creative design / decoding step** using the modified test script.

### 1️⃣ Example: run `test_copy.py` for sequence extraction

```bash
CUDA_VISIBLE_DEVICES=0 python test_copy.py \
  --model /home/donut/YG_BABO/shape2prog/model/ckpts_program_generator_828/ckpt_epoch_40.t7 \
  --data  /home/donut/YG_BABO/shape2prog/data_test/test/data.h5 \
  --batch_size 64 \
  --save_path ./output/yg_test/ \
  --save_prog \
  --save_img
```bash

### 2️⃣ Outputs

After running the command above, the following folders are created under `--save_path`:

📁 programs/
└─ Decoded DSL programs inferred from the model
e.g., 0.txt, 1.txt, ...

📁 rhino_commands/
└─ Rhino-style modeling command sequences converted from DSL
e.g., 0_rhino.txt, ...

📁 images/
└─ Single-view rendered images of the reconstructed 3D shapes

📁 sequence_images/
└─ Per-step sequence execution screenshots from execute_shape_program_with_trace
e.g., sample_0/step_0.png, sample_0/step_1.png, ...


These files together form the **modeling sequence dataset**, which can be used for further:

✔ analysis  
✔ visualization  
✔ workflow modeling research  
✔ UI / modeling support systems

