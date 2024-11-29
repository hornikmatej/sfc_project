#!/bin/bash

# Define variables
LOGIN="xhorni20"
N="15"
ZIP_FILE="${N}-${LOGIN}.zip"
MAX_SIZE=2048  # 2 MB in KB

# Create the zip file with specified files
zip -r $ZIP_FILE poetry.toml pyproject.toml poetry.lock README.md sample.py helpers.py model.py doc/sfc_project.pdf

# Check the size of the zip file
FILE_SIZE=$(du -k "$ZIP_FILE" | cut -f1)

if [ $FILE_SIZE -gt $MAX_SIZE ]; then
  echo "The zip file exceeds the 2 MB limit. Please reduce the file size or provide a download link."
else
  echo "The zip file has been created successfully: $ZIP_FILE"
fi