#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:27:06 2020

@author: 
"""

prompt_text = "Initiating process, loading data etc. One or more prompt(s) \
will appear shortly, requiring user-specified input. Please wait..."
print(prompt_text)

exec(open("./text_mining_import_libraries.py").read())
exec(open("./text_mining_custom_functions_and_classes.py").read())
exec(open("./text_mining_load_and_prepare_data.py").read())
exec(open("./text_mining_prepare_and_fit_pipeline.py").read())
exec(open("./text_mining_model_performance.py").read())