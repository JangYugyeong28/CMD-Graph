#!/bin/bash

###############################################
# CMD-Graph voxel + program generation pipeline
# (Edit YOUR paths below before running)
###############################################

# === 0) USER CONFIG ==========================
MATLAB_PATH="<path_to_matlab>/bin/matlab"                 # e.g., /usr/local/MATLAB/R2023a/bin/matlab
WORKDIR="<path_to_shape2prog_root>"                       # e.g., ~/projects/shape2prog
MODEL_PATH="<path_to_checkpoint>.t7"                      # e.g., ./model/ckpts/epoch_40.t7
DATA_PATH="<path_to_data>.h5"                             # e.g., ./data_test/test/data.h5
OUTPUT_DIR="$WORKDIR/output/yg_test"                      # default out folder
CONDA_ENV_NAME="shapeenv"                                 # name of your conda environment
###############################################


echo "🚀 Running voxel generation pipeline..."
cd "$WORKDIR" || exit 1


# === 1) MATLAB voxel generation =================
echo "🔧 Running MATLAB (generate_voxels.m)..."
$MATLAB_PATH -nodisplay -nosplash -r "try, generate_voxels, catch, exit(1), end, exit(0);" \
    > log_generate_voxels.txt 2>&1

if [ $? -eq 0 ]; then
  echo "✔ MATLAB execution complete."
else
  echo "❌ MATLAB script failed."
  exit 1
fi


# === 2) Archive previous outputs =================
if [ -d "$OUTPUT_DIR" ]; then
    TS=$(date +"%Y%m%d_%H%M%S")
    mv "$OUTPUT_DIR" "${OUTPUT_DIR}_backup_$TS"
    echo "📦 Archived previous results → ${OUTPUT_DIR}_backup_$TS"
fi

mkdir -p "$OUTPUT_DIR"


# === 3) Activate environment =====================
echo "📌 Activating conda: $CONDA_ENV_NAME"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"


# === 4) Run program extraction ====================
echo "📄 Decoding shapes → DSL + Commands + Images..."
CUDA_VISIBLE_DEVICES=0 python test_copy.py \
    --model "$MODEL_PATH" \
    --data "$DATA_PATH" \
    --batch_size 64 \
    --save_path "$OUTPUT_DIR" \
    --save_prog \
    --save_img

echo "🎉 Done. Files are stored in:  $OUTPUT_DIR"
