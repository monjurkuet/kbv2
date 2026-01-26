#!/bin/bash
cd /home/muham/development/kbv2
echo "Installing unstructured with markdown support..."
pip3 install "unstructured[md]" "unstructured" > /dev/null 2>&1
pip3 install "markdown" "nltk" > /dev/null 2>&1
python3 -c "import nltk; nltk.download('punkt', quiet=True)" > /dev/null 2>&1
echo "Dependencies installed!"
