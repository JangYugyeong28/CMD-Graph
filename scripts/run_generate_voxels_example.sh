#!/bin/bash

##############################################
# Example pipeline for:
#   1) voxel generation (MATLAB, optional)
#   2) shape2prog test (program generation)
#   3) CMD-Graph post-processing (our code)
#   4) zipping results
#
# 👉 You MUST edit paths and environment names
#    to match your local setup.
##############################################

# --- User-specific paths (edit these) ---------------------

# Root directory of shape2prog
SHAPE2PROG_ROOT="/path/to/shape2prog"

# Path to MATLAB executable (if using generate_voxels.m)
MATLAB_PATH="$SHAPE2PROG_ROOT/MATLAB/bin/matlab"

# Output directory for shape2prog test results
YG_TEST_DIR="$SHAPE2PROG_ROOT/output/yg_test"

# Conda environment name
CONDA_ENV_NAME="shapeenv"

# Model checkpoint & data used by shape2prog test
MODEL_CKPT="$SHAPE2PROG_ROOT/model/ckpts_program_generator_828/ckpt_epoch_40.t7"
DATA_H5="$SHAPE2PROG_ROOT/data_test/test/data.h5"

# Our post-processing script (CMD-Graph)
WGRAPH_SCRIPT="$SHAPE2PROG_ROOT/wgraph_new_copy.py"

# ----------------------------------------------------------


######## 1) (Optional) MATLAB voxel generation #############

# Uncomment if you actually use MATLAB to generate voxels
# "$MATLAB_PATH" -nodisplay -nosplash -r \
#   "try, generate_voxels, catch, exit(1), end, exit(0);" \
#   > log_generate_voxels.txt 2>&1
#
# if [ $? -ne 0 ]; then
#   echo "Error: MATLAB script failed to execute."
#   exit 1
# fi


######## 2) Backup previous yg_test ########################

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -d "$YG_TEST_DIR" ]; then
  mv "$YG_TEST_DIR" "${YG_TEST_DIR}_${TIMESTAMP}"
fi
mkdir -p "$YG_TEST_DIR"


######## 3) Activate conda env #############################

# Adjust this to your conda installation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"


######## 4) Run shape2prog test ############################

cd "$SHAPE2PROG_ROOT"

CUDA_VISIBLE_DEVICES=0 python test_copy.py \
  --model "$MODEL_CKPT" \
  --data "$DATA_H5" \
  --batch_size 64 \
  --save_path "$YG_TEST_DIR" \
  --save_prog \
  --save_img


######## 5) Run CMD-Graph generation #######################

python "$WGRAPH_SCRIPT"


######## 6) Zip yg_test folder #############################

# Remove previous zip if exists
rm -f yg_test.zip

# Zip current yg_test
zip -r yg_test.zip "$YG_TEST_DIR"

echo "Done. Zipped results: yg_test.zip"
