#!/usr/bin/env python
import csv
import sys
import operator
from BeautifulSoup import BeautifulSoup

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
    new_words = []
    #allowed_length = 40
    for word in text_array:
        # if len(word) > 3:
        new_words.append(word)

    # returns an array tuples [("word", count), ("new_word", count)]
    # sorted_words = sorted(word_map.items(), key=operator.itemgetter(1), reverse=True)
    # for word in new_words:
    #     count = 0
    #     while count < word[1]:
    #         new_words.append(word[0])
    #         count += 1

    # get rid of columns less than n and add padding for columns less than n
    # if len(new_words) <= allowed_length:
    #     while len(new_words) <= allowed_length:
    #         new_words.append("")

    #new_words = new_words[0:allowed_length]

    return ' '.join(new_words)



with open("{0}_result.csv".format(csvFile), 'w') as csvfile_write:
    fieldnames = ['score', "title", "text"]
    writer = csv.DictWriter(csvfile_write, fieldnames=fieldnames)
    writer.writeheader()
    #wrote_head = False

    with open(csvFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            review = row[0]
            title = row[2]
            text = row[3]
            soup_text = BeautifulSoup(text)
            soup_title = BeautifulSoup(title)
            title = ''.join(soup_title.findAll(text=True)).encode('utf-8').strip()
            text = ''.join(soup_text.findAll(text=True)).encode('utf-8').strip()
            #text_array = text.split(" ")
            # this is where we are going to extract the characters from the array blob and assign weighted values
            row_object = {}
            #weighted_text = get_weighted_text(text_array)
            if row[0] == "5.0":
                row_object = {'score': 1}
            elif row[0] == "1.0" or polar == True:
                row_object = {'score': 0}

            # for index, value in enumerate(weighted_text):
            #     word_key = "word_{0}".format(index)
            #     value_key = "value_{0}".format(index)
                # if wrote_head == False:
                #     fieldnames.append('word_{0}'.format(index))
                    #fieldnames.append('value_{0}'.format(index))
            row_object["title"] = title
            row_object["text"] = text
                #row_object[value_key] = values[1]

            # if wrote_head == False:
            #     writer.writeheader()
            #     wrote_head = True

            writer.writerow(row_object)
