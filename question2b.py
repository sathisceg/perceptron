

import numpy as np
import sys

def leastSquareApproach(inputdata):
    size = len(inputData)

    b = [1 for x in range(size)]

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for row in inputData:
        if (row[0] == 1):
            x1.append(row[1])
            y1.append(row[2])
        else:
            x2.append(row[1])
            y2.append(row[2])

    b_transpose = np.matrix(b).getT()
    matrix_a = np.matrix(inputData)
    matrix_a_transpose = matrix_a.getT()

    a_transpose_a = matrix_a_transpose * matrix_a
    a_transpose_a_inverse = a_transpose_a.getI()
    matrix_x = a_transpose_a_inverse * matrix_a_transpose
    result = matrix_x * b_transpose

    w0 = result.flat[0]
    w1 = result.flat[1]
    w2 = result.flat[2]

    lx=x1+x2
    # print(lx)
    ly=[]

    for i in lx:
        t = -(w0 + w1 * i) / w2
        ly.append(t)

    import matplotlib.pyplot as plt

    plt.plot(lx, ly)
    plt.scatter(x1, y1, label="c1", color="red", marker="*")
    plt.scatter(x2, y2, label="c2", color="blue", marker="*")
    plt.suptitle('least square approach', fontsize=12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("graph")
    plt.legend()
    plt.show()

    ####################################################################################################################

def lda(inputdata):
    class1 = []
    class2 = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for row in inputData:
        if (row[0] == 1):
            class1.append(row)
            x1.append(row[1])
            y1.append(row[2])
        else:
            class2.append(row)
            x2.append(row[1])
            y2.append(row[2])

    class1_matrix = np.matrix(class1)
    class2_matrix = np.matrix(class2)

    mclass1_matrix = class1_matrix[:, 1:3]
    mclass2_matrix = class2_matrix[:, 1:3]
    tmc1 = mclass1_matrix - mclass1_matrix.mean(axis=0)
    tmc2 = mclass2_matrix - mclass2_matrix.mean(axis=0)

    cov_c1 = tmc1.getT() * tmc1
    cov_c2 = tmc2.getT() * tmc2
    Sw = cov_c1 + cov_c2

    inverse_of_Sw = Sw.getI()

    mean_of_matrix = mclass1_matrix.mean(axis=0) - mclass2_matrix.mean(axis=0)
    result = inverse_of_Sw * mean_of_matrix.getT()

    w0 = result.flat[0]
    w1 = result.flat[1]
    lx = x1 + x2
    ly = []

    for i in lx:
        t = (w0 + w1 * i)
        ly.append(t)

    import matplotlib.pyplot as plt

    plt.plot(lx, ly)
    plt.scatter(x1, y1, label="c1", color="red", marker="*")
    plt.scatter(x2, y2, label="c2", color="blue", marker="*")
    plt.suptitle('linear discriminant analysis', fontsize=12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("graph")
    plt.legend()
    plt.show()






if __name__=='__main__':

    inputData = [[1, 3, 3], [1, 3, 0], [1, 2, 1], [1, 0, 1.5], [-1, -1, 1], [-1, 0, 0], [-1, -1, -1], [-1, 1, 0]]

    if(sys.argv[1]=='1'):
         leastSquareApproach(inputData)
    elif(sys.argv[1]=='2'):
        lda(inputData)
    else:
        print("1.leastSquareApproach ")
        print("2.linear discriminant analysis")

