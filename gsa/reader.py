import csv


class ReadFromCSV(object):
    def __init__(self, file_name, coding_dict={}):
        self.file_name = file_name
        self.coding_dict = coding_dict

    def from_file_to_data_list(self, file_name=None, label_name='label'):
        if file_name is None:
            file_name = self.file_name
        # csvfile = open('/Users/danil.gizdatullin/Documents/folder/Kaggle/new data exp/Men_dataset.csv', 'r')
        csvfile = open(file_name, 'rU')

        sreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        names = sreader.next()  # stores attributes' names

        data_list = []
        for row in sreader:
            # reads all lines in the csv file
            # and stores them in a dictionary with the value of an attribute

            data_dict = {}
            attr_ind = 0
            if row != ['', '', '', '', '', '', '', '']:
                for cell in row:
                    if cell != '':
                        data_dict[names[attr_ind]] = int(cell)
                    attr_ind += 1
                data_list.append(data_dict)
            # cnt+=1
            # if cnt==20: break
        csvfile.close()

        ds_men, ds_label = self._data_list_to_sequence_list2(data_list, label_name=label_name)

        return ds_men, ds_label

    def _data_list_to_sequence_list2(self, data_list, label_name='label'):
        # maps attributes to a sequence based on sorting them by age in an ascending order
        # taking into account equal ages

        sequence_list = []
        label_sequence = []
        # names=sorted(names)

        for row in data_list:
            temp_serq = sorted(row.keys(), key=row.get)
            sequence = []
            prev_ev = ''
            for ev in temp_serq:
                if ev != label_name:
                    if prev_ev == '':
                        sequence.append([str(self.coding_dict[ev])])
                        # print row
                    elif row[prev_ev] == row[ev]:
                        # print row
                        sequence[-1].append(str(self.coding_dict[ev]))
                    else:
                        # print row
                        sequence.append([str(self.coding_dict[ev])])
                    prev_ev = ev
                else:
                    label_sequence.append(row[ev])
            sequence_list.append(sequence)

        return sequence_list, label_sequence

    # def _create_two_different_files(self, ):

if __name__ == '__main__':
    data = ReadFromCSV('/Users/danil.gizdatullin/git_projects/'
                             'Trie_for_sequential_rules/data_experiment/full_data_shuffle.csv',
                             coding_dict={'work': 1,
                                          'separation': 2,
                                          'partner': 3,
                                          'marriage': 4,
                                          'children': 5,
                                          'parting': 6,
                                          'divorce': 7,
                                          'education': 8})
    data, label = data.from_file_to_data_list(label_name='label')
    print(data)
    print(label)

    print(len(data))
    print(len(label))
