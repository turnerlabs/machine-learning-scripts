#!/bin/bash

for filename in *.csv; do
    python csv_converter.py ${filename} $1
done
