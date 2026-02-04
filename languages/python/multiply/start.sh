python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
py.test
flake8

pip install mlx # if on Apple GPU
pip install cupy # if on NVIDIA GPU

multiply
# view jupyter notebook for figures

#nvidia-smi -L
# apple: system_profiler SPDisplaysDataType