# BASE_PATH is the location where you want the conda environment enstalled
# NEW_ENV is the name of the new conda environment you are building
BASE_PATH=/software/rbdgx
NEW_ENV=megatron-deepspeed

# Set these according to your sustem
export CUDA_BASE=/usr/local/cuda-11.3
export NCCL_BASE=/usr/local/nccl_2.9.6-1+cuda11.3_x86_64
export CUDNN_BASE=/usr/local/cuda-11.3
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_BASE
export NCCL_ROOT_DIR=$NCCL_BASE
export CUDNN_ROOT=$CUDNN_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

DATE=$(date +%Y-%m-%d)
mkdir $BASE_PATH/$DATE
pushd $BASE_PATH/$DATE
mkdir -p /$BASE_PATH/conda_env

conda create --prefix $BASE_PATH/conda_env/$NEW_ENV-${DATE} python=3.8
conda activate $BASE_PATH/conda_env/$NEW_ENV-${DATE}
conda install -y mpi4py
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses pytest matplotlib pandas

# This was incompatible with python 3.8
# conda install -y opencv # needed for pytorch image capability

PT_REPO_URL=https://github.com/pytorch/pytorch.git
PT_REPO_TAG="v1.10.0"

git clone --recursive $PT_REPO_URL
cd pytorch
git checkout --recurse-submodules $PT_REPO_TAG
python setup.py bdist_wheel
python setup.py install
cd ..
python -c 'import torch ; print(torch.__version__)'

# Installing apex
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd .. 
# python -c "import apex"

# Installing triton
# git clone https://github.com/openai/triton.git;
# cd triton/python;
# pip install cmake; # build time dependency
# pip install -e .

# installing gpt-neox
git clone https://github.com/EleutherAI/gpt-neox.git
cd gpt-neox
pip install -r requirements/requirements.txt
python prepare_data.py -d ./data
python ./deepy.py train.py -d configs small.yml local_setup.yml
cd ..

pushd
