import os
import random

class LITS_fix:
    def __init__(self, raw_dataset_path, fixed_dataset_path):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path

        if not os.path.exists(self.fixed_path):  # Create a save directory
            os.makedirs(self.fixed_path + 'data')
            os.makedirs(self.fixed_path + 'label')

        self.write_train_val_test_name_list()  # Create index txt file

    def write_train_val_test_name_list(self):
        train_rate = 0.6
        test_rate = 0.4

        assert test_rate + train_rate == 1.0

        c = 528  # the number of one brain data
        data_name_list_1, train_name_list_1 = self.split_data('brain1', train_rate, 0, c)

        data_name_list = []

        data_name_list.extend(data_name_list_1)

        random.shuffle(data_name_list)
        print('data :', len(data_name_list))

        self.write_name_list(data_name_list, "brain1.txt")

    def read_name_list(self, path, x, y):
        data_name_list = sorted(os.listdir(self.fixed_path + "/" + path))[x:y]
        for i in range(len(data_name_list)):
            data_name_list[i] = path + "/" + data_name_list[i]
        return data_name_list

    def write_name_list(self, name_list, file_name):
        f = open(self.fixed_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(str(name_list[i]) + "\n")
        f.close()

    def split_data(self, file_name, train_rate, x, y):
        data_name_list = self.read_name_list(file_name, x, y)
        random.shuffle(data_name_list)
        data_num = len(data_name_list)
        print(file_name, data_num)
        train_name_list = data_name_list[0:int(data_num * train_rate)]
        return data_name_list, train_name_list


def main():
    raw_dataset_path = 'train_dataset/'  # path of train data，train_dataset
    fixed_dataset_path = raw_dataset_path

    LITS_fix(raw_dataset_path, fixed_dataset_path)


if __name__ == '__main__':
    main()

