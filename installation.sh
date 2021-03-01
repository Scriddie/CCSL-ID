python -m venv ccslid_env
source ccslid_env/bin/activate
# Some debian pip issue (https://github.com/pypa/pip/issues/4823) requires:
pip install --ignore-installed --no-cache-dir --upgrade pip
pip install -r requirements.txt