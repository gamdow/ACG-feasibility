#!/bin/bash
# does jupyter exist?
if hash jupyter-notebook 2>/dev/null; then
  jupyter notebook --generate-config --allow-root
  echo "c.NotebookApp.iopub_data_rate_limit = 100000000" >> ~/.jupyter/jupyter_notebook_config.py
  mkdir ~/.jupyter/custom
  echo ".container { width:100% !important; }" > ~/.jupyter/custom/custom.css
  mkdir ~/.jupyter/nbconfig
  echo "{\"MarkdownCell\":{\"cm_config\":{\"lineWrapping\":true}},\"CodeCell\":{\"cm_config\":{\"lineWrapping\":true}}}" > ~/.jupyter/nbconfig/notebook.json
fi
