#!/usr/bin/env python
import csv
import sys

# arguments
# @args[1] csvFile: the csvFile to convert
# @args[2] polar: if present, then a polar opposite conversion is done
# example: python csv_converter.py file_name.csv polar
# example 2: python csv_converter.py file_name.csv
args = sys.argv

try:
    print("Converting {0}".format(args[1]))
    csvFile = args[1]
except e:
    print "Must pass in csv to convert"
    sys.exit()

polar = True
try:
    polar = args[2]
    polar = True
except:
    polar = False


with open("{0}_result.csv".format(csvFile), 'w') as csvfile_write:
    fieldnames = ['score', 'text']
    writer = csv.DictWriter(csvfile_write, fieldnames=fieldnames)
    writer.writeheader()

    with open(csvFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            review = row[0]
            text = row[3].replace('<span class="a-size-base review-text">', '').replace('</span>', '')
            if row[0] == "5.0":
                writer.writerow({'score': 1, 'text': text})
            elif row[0] == "1.0" or polar == True:
                writer.writerow({'score': 0, 'text': text})
