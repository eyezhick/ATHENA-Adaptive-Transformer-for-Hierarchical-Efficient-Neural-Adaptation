#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Generate the diagram
python architecture.py

echo "Architecture diagram generated at docs/architecture.png" 