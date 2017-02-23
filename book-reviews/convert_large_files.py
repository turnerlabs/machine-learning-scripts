import gzip
import sys
import csv

args = sys.argv
csvFile = ""

def parse_file(path):
  print path
  g = gzip.open(path, 'r')
  with open("{0}_result.csv".format(path), 'w') as csvfile_write:
      fieldnames = ['score', "label", "text"]
      writer = csv.DictWriter(csvfile_write, fieldnames=fieldnames)

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
            row["text"] = data["reviewText"]
            writer.writerow(row)

# try:
print("Converting {0}".format(args[1]))
csvFile = args[1]
parse_file(csvFile)
# except e:
#     print "Must pass in csv to convert"
#     sys.exit()
