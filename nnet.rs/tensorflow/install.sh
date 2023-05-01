#!/usr/bin/env bash
virtualenv .env --no-setuptools --no-wheel
source .env/bin/activate
pip install ~/.local/tensorflow/
