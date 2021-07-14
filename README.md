# Generating Sphinx documentation for pxtextmining

This branch is for building this HTML:

https://cdu-data-science-team.github.io/pxtextmining/index.html

Below steps based on 
[this](https://fazlerabbi37.github.io/blogs/publish_sphinx_doc_with_github_pages.html)
plus much of personal experimentation.

Run `python setup.py install`;

Create a docs folder or empty an existing one;

In docs folder run `sphinx-quickstart` (separate build and source);

When done, replace the contents in `docs/source/conf.py` with:

```
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../pxtextmining'))


# -- Project information -----------------------------------------------------

project = 'pxtextmining'
copyright = '2021, NHS England'
author = 'Andreas D Soteriades'

# The full version, including alpha/beta/rc tags
release = '0.2.6'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
```
**DON'T FORGET TO UPDATE THE VERSION NUMBER IN THE ABOVE TEXT!**

In `docs/Makefile` replace `BUILDDIR      = build` with `BUILDDIR      = ..`;

In `docs/make.bat` replace `set BUILDDIR=build` with
```
set BUILDDIR=..
REM set BUILDDIR=build
```

Rename docs folder to "html";

In html folder run:
```
sphinx-apidoc -o ./source ../pxtextmining
make html
```

Create a `.nojekyll` file (`touch .nojekyll`) in html;

Rename html to docs;
