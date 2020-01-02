import pickle
import os
import csv


def gen_csv(filename):
    data_dict = pickle.load(open(filename,'rb'))
    file_name = os.path.splitext(filename)[0]
    csv = open(file_name+"_labeled.csv","w+")
    for name,text in data_dict.items():
        csv.write(name + ',*START*\n')
        # remove all commas from string so they don't mess up the csv
        text = text.replace(",","")
        split_words = text.split()
        for word in split_words:
            csv.write(word + "," + "O" + "\n")
        csv.write("None" + ",*END*\n")
    csv.close()

def read_csv(filename):
    csv_file = open(filename,"r")
    read_csv = csv.reader(csv_file,delimiter=",")
    result_dict = {}
    curr_name = None
    for row in read_csv:
        word,label = row
        if label == "*START*":
            curr_name = word
            result_dict[curr_name] = []
        elif label == "*END*":
            curr_name = None
        else:
            result_dict[curr_name].append((word,label))
    return result_dict

if __name__ == "__main__":
    print(read_csv("dev_data_labeled.csv"))
