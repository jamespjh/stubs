python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
py.test
flake8

nvidia-smi -L
# apple: system_profiler SPDisplaysDataType