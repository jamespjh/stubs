python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
py.test
flake8
stubpy
