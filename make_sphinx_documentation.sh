#!/bin/bash

# Once changes to docstrings have been saved, go to the root directory

# Delete all .rst files except for index.rst from docs/source
cd docs/source
find ./ -name "*.rst" -not -name "index.rst"  -exec rm {} \;

python setup.py install

# In docs
cd docs

sphinx-apidoc -o ./source ../pxtextmining
make clean
make html

# Code below from:
# https://github.com/fazlerabbi37/Code.random/blob/c7ae5ec32a8b6eb703a37cd6085a557f503a856c/shell/make_github_pages_from_sphinx.sh
# https://fazlerabbi37.github.io/blogs/publish_sphinx_doc_with_github_pages.html

#rename docs to html
mv docs html

#change directory to html
cd html

#make documents
make html

#create .nojekyll file
touch .nojekyll

#change directory back to root dir
cd ..

#rename html to docs
mv html docs