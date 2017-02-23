import gzip
import sys
import csv

args = sys.argv
csvFile = ""
allowed_length = 100

def get_weighted_text(text_array):
    new_words = []
    for word in text_array:
        #if len(word) > 3:
        new_words.append(word)

    # returns an array tuples [("word", count), ("new_word", count)]
    # sorted_words = sorted(word_map.items(), key=operator.itemgetter(1), reverse=True)
    # for word in new_words:
    #     count = 0
    #     while count < word[1]:
    #         new_words.append(word[0])
    #         count += 1

    # get rid of columns less than n and add padding for columns less than n
    if len(new_words) <= allowed_length:
        while len(new_words) <= allowed_length:
            new_words.append("")

    new_words = new_words[0:allowed_length]

    return new_words

def parse_file(path):
  print path
  g = gzip.open(path, 'r')
  with open("{0}_result.csv".format(path), 'w') as csvfile_write:
      fieldnames = ['score']
      write = False
      wrote_head = False

      for l in g:
        data = eval(l)
        score = ""
        if data["overall"] == 1:
            score = 0
        elif data["overall"] == 5:
            score = 1

        if score == 0 or score == 1:
            row = {}
            row["score"] = score
            split_text = data["reviewText"].split(" ")
            new_text = get_weighted_text(split_text)

            for index,value in enumerate(new_text):
                key = "value_{0}".format(index)
                fieldnames.append(key)
                row[key] = value

            if wrote_head == False:
                writer = csv.DictWriter(csvfile_write, fieldnames=fieldnames)

            writer.writerow(row)

# try:
print("Converting {0}".format(args[1]))
csvFile = args[1]
parse_file(csvFile)
# except e:
#     print "Must pass in csv to convert"
#     sys.exit()
