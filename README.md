# Generating Command Modeling and Design Graphs with Data Augmentation for Enhanced 3D Modeling Support

<img width="595" height="250" alt="Figure1 (1)" src="https://github.com/user-attachments/assets/87d7da6a-f9e2-4849-bb5f-df9caca59d6e" />

## 
[[ScienceDirect]](https://www.sciencedirect.com/science/article/abs/pii/S1474034625005373) <br>
[Yugyeong Jang](https://yugyeong.cargo.site/), [Kyung Hoon Hyun](https://designinformatics.hanyang.ac.kr/People_Kyung-Hoon-Hyun)

---

## 🔍 Overview

This repository contains the **implementation for inferring 3D modeling command sequences from 3D shapes** and converting them into **Command Modeling and Design Graphs (CMD-Graphs)**, as proposed in our paper:

> *“Generating Command Modeling and Design Graphs with Data Augmentation for Enhanced 3D Modeling Support”* (Advanced Engineering Informatics, 2025)

Modern 3D modeling tools often impose a high cognitive load, especially for beginners, because the **underlying modeling sequence is hidden** behind the final 3D shape.  
Our goal is to **reconstruct that hidden sequence** from completed 3D models and turn it into a **graph structure** that can support:

- Understanding how a model was built  
- Suggesting alternative modeling paths  
- Providing command-level assistance during modeling

This repository focuses on the **modeling sequence extraction pipeline**, not on general-purpose shape generation.

---

## 🗂 What This Repository Provides

This repo includes:

- ✅ **Code for modeling-sequence inference**
  - From voxelized shapes / OBJ files to a **domain-specific language (DSL)** representation
  - From DSL sequences to **Command Modeling and Design Graphs (CMD-Graphs)**

- ✅ **Data processing utilities**
  - Scripts for converting voxel data and 3D meshes (OBJ) into the internal representation used for sequence inference
  - Tools for handling **repetitions, symmetry, and structural grouping** in 3D shapes

- ✅ **Example configurations / demo scripts**
  - Example scripts that show how to:
    - Take an input OBJ file
    - Run sequence inference
    - Export the resulting CMD-Graph (e.g., as JSON)

This repository **does NOT** provide:

- ❌ Full training code for the original shape generation model  
- ❌ The original training dataset itself (we build on datasets from prior work; see below)  
- ❌ General-purpose 3D reconstruction from images or sketches

Instead, this repo is focused on:

> **“Given 3D shapes (from a pre-trained model + dataset), how do we infer the modeling command sequence and encode it as a graph?”**

---

## 📦 Upstream Dataset & Training (shape2prog)

For training the underlying model and generating 3D shapes, we build on:

- **shape2prog (Huang et al.)**  
  GitHub: https://github.com/HobbitLong/shape2prog  

In our work:

- We use the **dataset configuration and pipeline proposed in shape2prog**.
- The voxel representations are **upsampled to 64×64×64** resolution for training.
- The trained model is then used to:
  - Generate or reconstruct 3D shapes
  - Provide inputs for our **modeling sequence inference pipeline**

Because of licensing and size constraints, **we do not re-distribute the original datasets or full training code** from shape2prog in this repository.  
Please refer to the original project for:

- Dataset preparation
- Training scripts
- Licensing and usage conditions

---

## 🧩 Modeling Sequence Inference

Our sequence inference pipeline proceeds in three stages:

1. **Geometry & Structure Reconstruction**
   - Reconstructs geometric elements and their structural relationships from voxel / mesh input
   - Uses a **domain-specific language (DSL)** to encode primitives and operations

2. **Command Sequence Inference**
   - Infers a plausible **modeling command sequence** that could have produced the final shape
   - Efficiently handles:
     - Repetitions
     - Symmetry
     - Hierarchical structures

3. **CMD-Graph Construction**
   - Converts the inferred sequence into a **workflow graph**
   - Nodes: intermediate 3D modeling states  
   - Edges: modeling commands with parameters  
   - This yields **richer, more detailed sequence data** than typical binary voxel datasets

In this repository, you will find:

- Scripts for **running sequence inference** on:
  - Shapes generated from the trained model (using shape2prog data)
  - Your own **OBJ files** (subject to preprocessing constraints)
- Utilities for **exporting CMD-Graphs** for downstream use (e.g., visualization, analysis, UI integration)

---

## 🚀 Getting Started (High-Level)

> 🔧 Note: The exact file names and paths may differ depending on how you organize your project.  
> Treat the commands below as a **template** and adapt them to your setup.

1. **Prepare Data (from shape2prog / your own shapes)**  
   - Follow the instructions in `shape2prog` to generate or reconstruct voxelized 3D shapes.
   - Upsample or convert them to **64×64×64** if needed.
   - Export meshes (OBJ) if you want to run sequence inference on mesh inputs.

2. **Run Modeling Sequence Inference**
   - Use the provided scripts (e.g., `scripts/infer_sequence_from_obj.py`) to:
     - Load an OBJ file
     - Convert it to the internal DSL representation
     - Infer the modeling sequence

3. **Export CMD-Graph**
   - The inferred sequence can be exported as:
     - A **graph JSON** (nodes + edges)
     - A sequence file for further training or analysis

Example (pseudo):

```bash
# Example: infer command sequence from an OBJ file
python scripts/infer_sequence_from_obj.py \
    --input_obj path/to/input/model.obj \
    --output_json path/to/output/cmd_graph.json
