from usa.reader import read_from_csv

sequence_reader = read_from_csv.SequencesFromFile(file_name='./data/full_data_shuffle.csv',
                                                  coding_dict={'work': 1,
                                                               'separation': 2,
                                                               'partner': 3,
                                                               'marriage': 4,
                                                               'children': 5,
                                                               'parting': 6,
                                                               'divorce': 7,
                                                               'education': 8}
                                                  )
data, label = sequence_reader.from_file_to_data_list(label_name='label')

print(data)
print(label)

print(len(data))
print(len(label))
