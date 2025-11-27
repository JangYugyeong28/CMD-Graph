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

## 📁 Repository Structure
cmd-graphs/

├── test.py                         # main decoding script (voxel → DSL → images)

└── scripts/

    ├── run_generate_voxels_example.sh   # voxel generation & automated decoding pipeline
    
└── README.md                        # usage notes for scripts/

---

## 📁 What This Repository Provides
| Included                                   | Description                                          |
| ------------------------------------------ | ---------------------------------------------------- |
| **test.py**                                | Extracts DSL, Rhino commands, images, sequence steps |
| **scripts/run_generate_voxels_example.sh** | Full example pipeline script                         |
| **Images + step exports**                  | Chunked output for each modeling execution           |
❌ Training pipeline / dataset are not included
❌ CMD-Graph generation layer is not part of this release

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

## 🚀 Usage — Modeling Command Extraction

### **1) Direct Python decoding (recommended)**

```bash
# Example execution (edit paths before running)

CUDA_VISIBLE_DEVICES=0 python test.py \
  --model <path_to_checkpoint>.t7 \        # ex) ./model/ckpt_epoch_40.t7
  --data  <path_to_test_data>.h5 \         # ex) ./data/test/data.h5
  --batch_size 64 \
  --save_path <output_dir>/ \              # ex) ./output/run_01/
  --save_prog \                            # export DSL programs
  --save_img                               # export mesh + sequence screenshots
```

### **2) Full automated pipeline (.sh script)**
```bash
scripts/run_generate_voxels_example.sh
```
✔ runs MATLAB → voxel generation

✔ backs up old results automatically

✔ activates conda + executes decoding end-to-end

⚠ Make sure to update paths inside the script before use.

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

---

## 📚 Citation

If you use this code or reference our modeling sequence extraction approach, please cite:

```bibtex
@article{jang2025generating,
  title={Generating command modeling and design graphs with data augmentation for enhanced 3D modeling support},
  author={Jang, Yugyeong and Hyun, Kyung Hoon},
  journal={Advanced Engineering Informatics},
  volume={68},
  pages={103644},
  year={2025},
  publisher={Elsevier}
}
