#!/usr/bin/env bash
# GMR (General Motion Retargeting) 설치
set -e

SRC=/home/sehoon/Projects/GMR

if [ ! -d "$SRC" ]; then
  echo "[1/4] cloning GMR"
  git clone https://github.com/YanjieZe/GMR.git "$SRC"
else
  echo "[1/4] GMR already cloned, skipping"
fi

source /home/sehoon/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^gmr "; then
  echo "[2/4] creating conda env: gmr (python 3.10)"
  conda create -n gmr python=3.10 -y
else
  echo "[2/4] env gmr exists, skipping"
fi

conda activate gmr

echo "[3/4] installing GMR"
cd "$SRC"
pip install -e .

echo "[4/4] fixing libstdc++ for mujoco viewer"
conda install -c conda-forge libstdcxx-ng -y

echo
echo "done. usage:"
echo "  conda activate gmr"
python -c "import general_motion_retargeting; print('import ok')" || true
