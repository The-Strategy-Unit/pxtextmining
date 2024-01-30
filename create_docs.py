import glob
import os

"""
Python script to automatically generate .md files in docs/reference based on
contents of pxtextmining folders
"""


modules = glob.glob('pxtextmining/*/')
module_names = []
for folder in modules:
    if '__' not in folder:
        module_name = folder.split('/')[-2]
        print(f'MODULE: {module_name}')
        pylist = glob.glob(f"{folder}/*.py")
        for py in pylist:
            if '__' not in py:
                py_name = os.path.basename(py)[:-3]
                print(py_name)
                with open(f'docs/reference/{module_name}/{py_name}.md', 'w') as f:
                    if module_name == 'helpers':
                        f.write(f"""::: pxtextmining.{module_name}.{py_name}
    options:
        show_source: true""")
                    else:
                        f.write(f'::: pxtextmining.{module_name}.{py_name}')
