import numpy as np
import pandas as pd

from entropy_topsis_small import standarize
from from_small_to_big import handle_null_values, read_sheet

need_topsis = False

topsis_path = "output_topsis.xlsx"
topsis = pd.read_excel(topsis_path, header=0)
layer1_path = "output_weights_small.xlsx"
layer1 = pd.read_excel(layer1_path, header=0)
layer2_path = "output_weights_large.xlsx"
layer2 = pd.read_excel(layer2_path, header=0)
layer3_path = "output_weights_final.xlsx"
layer3 = pd.read_excel(layer3_path, header=0)

data_path = "/Users/stone/Downloads/olympic_data/train_data.xlsx"
# data = pd.read_excel(data_path, header=0)


def f(x, debug=False):
    # 2 * 13

    a1 = x[0] * layer1.iloc[:, 1]
    a2 = x[1] * layer1.iloc[:, 1]
    b = np.zeros(12)
    b[0] = a1[0] + a1[1] + a1[2]
    b[1] = a1[3] + a1[4]
    b[2] = a1[5] + a1[6]
    b[3] = a1[7]
    b[4] = a1[8] + a1[9] + a1[10]
    b[5] = a1[11] + a1[12]
    b[6] = a2[0] + a2[1] + a2[2]
    b[7] = a2[3] + a2[4]
    b[8] = a2[5] + a2[6]
    b[9] = a2[7]
    b[10] = a2[8] + a2[9] + a2[10]
    b[11] = a2[11] + a2[12]

    c = np.zeros(4)
    c[0] = np.sum(b * layer2.iloc[:, 1])
    c[1] = np.sum(b * layer2.iloc[:, 2])
    c[2] = np.sum(b * layer2.iloc[:, 3])
    c[3] = np.sum(b * layer2.iloc[:, 4])

    d = np.sum(c * layer3.iloc[:, 1])
    if debug:
        # print("a1:", a1)
        # print("a2:", a2)
        print("b:", b)
        print("c:", c)
        print("d:", d)
    return b, c, d


def fff(subject_label, year):
    a = np.zeros((2, 13))
    page1 = pd.read_excel(data_path, sheet_name="Popularity and Accessibility", header=0)
    # page1 = read_sheet(0)
    s1 = standarize(handle_null_values(page1.iloc[:30, 1:8]))
    s2 = standarize(handle_null_values(page1.iloc[:30, 9:16]))
    s3 = standarize(handle_null_values(page1.iloc[:30, 17:25]))
    # print(s1)
    a[0, 0] = s1.iloc[subject_label, year]
    # print(subject_label, a[0][0])
    a[0][1] = s2.iloc[subject_label, year]
    a[0][2] = s3.iloc[subject_label, year]
    a[1][0] = s1.iloc[subject_label, 1+year]
    a[1][1] = s2.iloc[subject_label, 1+year]
    a[1][2] = s3.iloc[subject_label, 1+year]
    page2 = pd.read_excel(data_path, sheet_name="Gender Equity", header=0)
    # page2 = read_sheet(1)
    s1 = standarize(handle_null_values(page2.iloc[:30, 1:8]))
    s2 = standarize(handle_null_values(page2.iloc[:30, 9:16]))
    a[0][3] = s1.iloc[subject_label, year]
    a[0][4] = s2.iloc[subject_label, year]
    a[1][3] = s1.iloc[subject_label, 1+year]
    a[1][4] = s2.iloc[subject_label, 1+year]
    page3 = pd.read_excel(data_path, sheet_name="Sustainability", header=0)
    s1 = standarize(handle_null_values(page3.iloc[:30, 1:8]))
    s2 = standarize(handle_null_values(page3.iloc[:30, 9:16]))
    a[0][5] = s1.iloc[subject_label, year]
    a[1][5] = s1.iloc[subject_label, 1+year]
    a[0][6] = s2.iloc[subject_label, year]
    a[1][6] = s2.iloc[subject_label, 1+year]
    page4 = pd.read_excel(data_path, sheet_name="Inclusivity", header=0)
    s1 = standarize(handle_null_values(page4.iloc[:30, 1:8]))
    a[0][7] = s1.iloc[subject_label, year]
    a[1][7] = s2.iloc[subject_label, 1+year]
    page5 = pd.read_excel(data_path, sheet_name="Relevance and Innovation", header=0)
    s1 = standarize(handle_null_values(page5.iloc[:30, 1:8]))
    s2 = standarize(handle_null_values(page5.iloc[:30, 9:16]))
    s3 = standarize(handle_null_values(page5.iloc[:30, 17:25]))
    a[0][8] = s1.iloc[subject_label, year]
    a[1][8] = s1.iloc[subject_label, 1+year]
    a[0][9] = s2.iloc[subject_label, year]
    a[1][9] = s2.iloc[subject_label, 1+year]
    a[0][10] = s3.iloc[subject_label, year]
    a[1][10] = s3.iloc[subject_label, 1+year]
    page6 = pd.read_excel(data_path, sheet_name="Safety and Fair Play", header=0)
    s1 = standarize(handle_null_values(page6.iloc[:30, 1:8]))
    s2 = standarize(handle_null_values(page6.iloc[:30, 9:16]))
    a[0][11] = s1.iloc[subject_label, year]
    a[1][11] = s1.iloc[subject_label, 1+year]
    a[0][12] = s2.iloc[subject_label, year]
    a[1][12] = s2.iloc[subject_label, 1+year]
    # print(a)
    return f(a)


def topsis_based_f(subject_label, year):
    x = subject_label + year * 36
    b = topsis.iloc[x, 1]


if __name__ == "__main__":
    x = np.random.rand(2, 13)  # 13 inputs in 2 years
    b = np.zeros(12)
    c = np.zeros(4)
    d = 0.
    # for i in range(114514):
    #     nb = f(x)
    #     b += nb[0]
    #     c += nb[1]
    #     d += nb[2]
    #     if i % 100 == 0:
    #         print("b:", b)
    #         print("c:", c)
    #         print("d:", d)
    sum = 0
    # for i in range(30):
    #     for j in range(6):
    #         # print()
    #         print(fff(i, j)[2])
    #         sum += fff(i, j)[2]
    # print(sum/30/6)
    # 23 5 12 2 20 22
    print(fff(21, 5)[2])
    print(fff(3, 5)[2])
    print(fff(10, 5)[2])
    print(fff(0, 5)[2])
    print(fff(18, 5)[2])
    print(fff(20, 5)[2])
