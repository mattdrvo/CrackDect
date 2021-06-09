# -- Project import -----------------------------------------------------------
from crackdect import *
# -- General configuration ---------------------------------------------------

needs_sphinx = '3.5'

extensions = ['sphinx.ext.autodoc',  # autodocumentation module
              # 'sphinx.ext.imgmath',  # mathematical expressions can be rendered as png images
              'numpydoc',           # docs in numpy-style
              'sphinx.ext.napoleon',  # numpy docstrings
              'sphinx.ext.intersphinx',
              'sphinx.ext.coverage',
              'sphinx.ext.autosummary',  # make autosummarys
              'sphinx.ext.viewcode',
              'sphinx.ext.autosectionlabel',
              # 'sphinx.ext.readthedocks'
              ]


templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

add_module_names = False

source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
html_show_sourcelink = True

# -- Render options ---------

# imgmath_image_format = 'svg'  # This would render mathematical expressions as svg images
imgmath_latex_preamble = r'\usepackage{xcolor}'  # mathematical expressions can be colored

# -- Options for HTML output ---------------------------------------------------

html_theme = 'sphinx_rtd_theme'


# -- autosummary settings -------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# numpydoc options
# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True

# numpydoc_show_class_members = True
# numpydoc_class_members_toctree = True
numpydoc_show_inherited_class_members = False

# -- Project information -----------------------------------------------------

project = 'CrackDect'
copyright = '2021, Matthias Drvoderic'
author = 'Matthias Drvoderic'
version = '0.1'
release = '0.1'