rem automatically update and upload ze package
python setup.py sdist bdist_wheel
python -m twine upload --repository agrisatpy dist/*
RMDIR "dist" /S /Q