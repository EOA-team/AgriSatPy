# bash script to install the package locally (without PyPi)

pip uninstall agrisatpy -y

python setup.py bdist_wheel
pip install dist/*
rm -rf dist/
