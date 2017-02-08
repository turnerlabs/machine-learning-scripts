#!/bin/bash

# call this in the same directory, which csv_converter is located in.
# make sure that all of your data files are located in the same folder as well
# this will convert every data file and write a _result.csv file with the
# crunched data
#
# params
# -f: the filename to convert, if left out then it will convert all files in the directory
# -p: whether or not pass in polar conversion
#
# example on how to call in docker
# docker: docker run -it -v book-review-data:/data  logistical-regression-converter -f /data/Suzanne-Collins-The-Hunger-Games.csv -p polar


while getopts ":f:p:" opt; do
  case $opt in
    f) filename="$OPTARG"
    ;;
    p) polar="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ -n "${filename}" ]; then
    python csv_converter.py ${filename} ${2}
else
    for filename in /data/*.csv; do
        python csv_converter.py ${filename} ${polar}
    done
fi
