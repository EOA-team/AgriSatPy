python setup.py sdist bdist_wheel
python -m twine upload --repository agrisatpy dist/*
rm -rf dist/

