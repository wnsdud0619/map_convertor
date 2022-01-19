#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

# --------------------------------------------------

def open_file(file_path, open_type):
        
    if not check_file(file_path):
        file = open(file_path, "w")
        file.close()

    file = open(file_path, open_type)

    return file

def write_file(file_name, msg):
    file = open_file(file_name, 'a')
    file.write(msg)

# --------------------------------------------------

def check_file(file_path):
    return os.path.exists(file_path)

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

