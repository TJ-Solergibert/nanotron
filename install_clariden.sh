# initial script to create the nanotron virtual environment.
# TODO(imanol): rely on req.txt file to be more reliable. 

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip setuptools wheel packaging
# pre installs
pip install torch

# actual requirements
pip install -r requirements.txt
pip install -e .
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPLUS_INCLUDE_PATH=/usr/include/python3.10/
