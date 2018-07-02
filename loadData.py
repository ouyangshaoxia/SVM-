import numpy as np

def get_data_target(file_name):
    data_mat_h = []
    data_mat_w = []
    label_mat = []
    fr = open(file_name)
    for line in fr:
        line_list = line.strip().split(' ')
        line_list_float = [float(data) for data in line_list]
        # data_mat_h.append(line_list_float[0:-2:2])
        # data_mat_w.append(line_list_float[0:-1:2])

        data_mat_h.append(line_list_float[0:-1])
        label_mat.append(line_list_float[-1])


    return data_mat_h, data_mat_w, label_mat


if __name__ == '__main__':
    data_mat_h, data_mat_w, label_mat = get_data_target('test_data.txt')
    count_i = 0
    for index in range(len(data_mat_h)):
        count_i += 1
        if len(data_mat_h[index]) != 10:
            print("index is :", index)
            print(data_mat_h[index])

    print(count_i)
