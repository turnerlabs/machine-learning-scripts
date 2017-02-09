#!/usr/bin/env python
import csv
import sys
import operator

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


def get_weighted_text(text_array):
    word_map = {}
    allowed_length = 5
    for word in text_array:
        if len(word) > 3:
            if not word_map.get(word):
                word_map[word] = 0

            word_map[word] += 1

    # returns an array tuples [("word", count), ("new_word", count)]
    sorted_words = sorted(word_map.items(), key=operator.itemgetter(1), reverse=True)
    # get rid of columns less than n and add padding for columns less than n
    if len(sorted_words) <= allowed_length:
        while len(sorted_words) <= allowed_length:
            sorted_words.append(("", 0))

    sorted_words = sorted_words[0:allowed_length]

    return sorted_words



with open("{0}_result.csv".format(csvFile), 'w') as csvfile_write:
    fieldnames = ['score', 'word_0', 'value_0', 'word_1', 'value_1', 'word_2', 'value_2', 'word_3', 'value_3', 'word_4', 'value_4']
    writer = csv.DictWriter(csvfile_write, fieldnames=fieldnames)
    writer.writeheader()

    with open(csvFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            review = row[0]
            text = row[3].replace('<span class="a-size-base review-text">', '').replace('</span>', '')
            text = "{0} {1}".format(row[2], text)
            text_array = text.split(" ")
            # this is where we are going to extract the characters from the array blob and assign weighted values
            weighted_text = get_weighted_text(text_array)
            if row[0] == "5.0":
                row_object = {'score': 1}
            elif row[0] == "1.0" or polar == True:
                row_object = {'score': 0}

            for index, values in enumerate(weighted_text):
                word_key = "word_{0}".format(index)
                value_key = "value_{0}".format(index)
                row_object[word_key] = values[0]
                row_object[value_key] = values[1]

            writer.writerow(row_object)
