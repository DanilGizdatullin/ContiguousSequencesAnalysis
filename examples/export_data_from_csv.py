from gsa.reader import ReadFromCSV

# create a object of class ReadFromCSV
# file_name is a path to file with data
# coding dict is a dict of column names to int numbers
sequence_reader = ReadFromCSV(file_name='./data/full_data_shuffle.csv',
                              coding_dict={'work': 1,
                                           'separation': 2,
                                           'partner': 3,
                                           'marriage': 4,
                                           'children': 5,
                                           'parting': 6,
                                           'divorce': 7,
                                           'education': 8})

# call from_file_to_data_list method to create two lists
# first list with data, second list with labels
data, label = sequence_reader.from_file_to_data_list(label_name='label')

print(data)
print(label)

print(len(data))
print(len(label))
