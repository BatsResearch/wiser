import pickle
import os
import csv

def dict_to_csv(file_path):
    data_dict = pickle.load(open(file_path, 'rb'))
    file_name = os.path.splitext(file_path)[0]

    with open(file_name + ".csv", 'w+') as file:
        writer = csv.writer(file)
        for actor, sentences in data_dict.items():
            writer.writerow(('*START-ACTOR*', actor))
            for sent in sentences:
                writer.writerow(('*START-SENTENCE*', actor))
                for token in sent:
                    writer.writerow(token)
                writer.writerow(('*END-SENTENCE*', actor))
            writer.writerow(('*END-ACTOR*', actor))


def csv_to_dict(file_path):
    file_name = os.path.splitext(file_path)[0]

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=",")
        result_dict = {}
        sentence = []
        for row in reader:
            word, label = row
            print(row)
            if label == 'i-MOV':
                label = 'I-MOV'
            if word == '\n':
                continue
            if word == "*START-ACTOR*":
                result_dict[label] = []
            elif word == "*START-SENTENCE*":
                sentence = []
            elif word == "*END-SENTENCE*":
                if len(sentence) > 1:
                    result_dict[label].append(sentence)
                sentence = []
            elif word == "*END-ACTOR*":
                pass
            else:
                sentence.append((word, label))

    with open(file_name + '.p', 'wb') as f:
        pickle.dump(result_dict, f)
