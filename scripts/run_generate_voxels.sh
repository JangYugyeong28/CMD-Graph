#!/bin/bash

# MATLAB 실행 경로 설정 (MATLAB이 설치된 경로로 수정하세요)
MATLAB_PATH="/home/donut/YG_BABO/shape2prog/MATLAB/bin/matlab"

# MATLAB 스크립트 실행 (생성된 로그는 log_generate_voxels.txt 파일에 저장)
$MATLAB_PATH -nodisplay -nosplash -r "try, generate_voxels, catch, exit(1), end, exit(0);" > log_generate_voxels.txt 2>&1

# MATLAB 실행 결과 확인
if [ $? -eq 0 ]; then
  echo "MATLAB script executed successfully."
else
  echo "Error: MATLAB script failed to execute."
fi

# yg_test 폴더의 이름을 현재 시간으로 변경 (백업)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mv /home/donut/YG_BABO/shape2prog/output/yg_test /home/donut/YG_BABO/shape2prog/output/yg_test_$TIMESTAMP

# 새로운 yg_test 폴더 생성
mkdir /home/donut/YG_BABO/shape2prog/output/yg_test

# conda 환경 활성화 (conda activate 사용)
source /home/donut/anaconda3/etc/profile.d/conda.sh
conda activate shapeenv

# 작업 디렉토리 설정
cd /home/donut/YG_BABO/shape2prog

# creative design 실행
CUDA_VISIBLE_DEVICES=0 python test_copy.py --model /home/donut/YG_BABO/shape2prog/model/ckpts_program_generator_828/ckpt_epoch_40.t7 --data /home/donut/YG_BABO/shape2prog/data_test/test/data.h5 --batch_size 64 --save_path ./output/yg_test/ --save_prog --save_img

# wgraph_new.py 실행
python /home/donut/YG_BABO/shape2prog/wgraph_new_copy.py

# 이전 yg_test.zip 파일 삭제
rm -rf yg_test.zip /home/donut/YG_BABO/shape2prog/yg_test.zip

# yg_test 폴더를 압축
zip -r yg_test.zip /home/donut/YG_BABO/shape2prog/output/yg_test
