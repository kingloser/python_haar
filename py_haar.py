import numpy as np
import _thread
x = [10, 13, 25, 26, 29, 21, 7, 15]
add = []
sub = []
length = len(x)
end = []
# 一维中的求和
y = [[120, 0, 0, 0, 0, 0, 0, 0],
     [20, 20, 0, 0, 0, 0, 0, 0],
     [60, 60, 63, 127, 127, 63, 0, 0],
     [70, 80, 127, 255, 255, 127, 0, 0],
     [0, 0, 127, 255, 255, 127, 0, 0],
     [0, 0, 63, 127, 127, 63, 0, 0],
     [0, 30, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]
     ]
print(y)


def add_sum(x, length):
    for i in range(0, length, 2):
        # print(i)
        z = (x[i] + x[i + 1])/2.0
    # i = i+1
        add.append(z)
    # for i in add:
    #     print(i)


def sub_sub(x, length):
    for i in range(0, length, 2):
        z = (x[i] - x[i + 1]) / 2.0
        sub.append(z)

    # for i in sub:
    #     print(i)
# try:
#     _thread.start_new_thread(add_sum, ("Thread-1", length))
#     _thread.start_new_thread(sub_sub, ("Thread-2", length))
# except:
#     print("Error: unable to start thread")
for i in y:
    print("uu", i)
    sub_sub(i, length)
    add_sum(i, length)
    end.append(add)
    end.append(sub)
    add = []
    sub = []
    # add = end
    # sub = end
# b1 = np.array()
# end = add + sub
# end = add
b1 = np.array(end)
b1 = b1.reshape((8, 8), order='c')
print('b1', b1)
end = []
b1 = np.transpose(b1)
for i in b1:
    # b1[:, i]
    sub_sub(i, length)
    add_sum(i, length)
    end.append(add)
    end.append(sub)
    add = []
    sub = []
b1 = np.array(end)
b1 = b1.reshape((8, 8), order='c')
b1 = np.transpose(b1)
b1 = b1*2
print('end:', b1)
# print(b1[:,1])

# for i in b1:
#     print(i)
