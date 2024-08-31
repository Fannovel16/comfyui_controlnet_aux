#!/bin/bash

requirements_txt="$(dirname "$0")/requirements.txt"
python_exec="../../../python_embeded/python.exe"

echo "Installing ComfyUI's ControlNet Auxiliary Preprocessors.."

if [ -f "$python_exec" ]; then
    echo "Installing with ComfyUI Portable"
    while IFS= read -r line; do
        "$python_exec" -s -m pip install "$line"
    done < "$requirements_txt"
else
    echo "Installing with system Python"
    while IFS= read -r line; do
        pip install "$line"
    done < "$requirements_txt"
fi
