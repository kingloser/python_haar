import numpy as np
import _thread
x = [10, 13, 25, 26, 29, 21, 7, 15]
add = []
sub = []
length = len(x)
# print(length)
end = []
# 一维中的求和
y = [[120, 0, 0, 0, 0, 0, 0, 0],
     [20, 20, 0, 0, 0, 0, 0, 0],
     [60, 60, 63, 127, 127, 63, 0, 0],
     [70, 80, 127, 255, 255, 127, 0, 0],
     [0, 0, 127, 255, 255, 127, 0, 0],
     [0, 0, 63, 127, 127, 63, 0, 0],
     [0, 30, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 2, 3, 4, 5, 6, 7, 8],
    #  [1, 2, 3, 4, 5, 6, 7, 8]
     ]
length = len(y)
weigth = len(y[0])
print(length)
print(weigth)
# b1 = np.array(y)

# b1 = b1.reshape(( length,weigth), order='c')
# print('b111111', b1) 
def add_sum(x, length):
    for i in range(0, length, 2):
        # print(i)
        z = (x[i] + x[i + 1])/2.0
    # i = i+1
        add.append(z)
def sub_sub(x, length):
    for i in range(0, length, 2):
        z = (x[i] - x[i + 1]) / 2.0

        sub.append(z)
#     print("Error: unable to start thread")
for i in y:
    print("uu", i)
    sub_sub(i, weigth)
    add_sum(i, weigth)
    end.append(add)
    end.append(sub)
    add = []
    sub = []
b1 = np.array(end)

b1 = b1.reshape((length,weigth), order='c')
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
end = end *2
print("end:",end)
b1 = b1.reshape((weigth,length), order='c')
b1 = np.transpose(b1)
b1 = b1 * 2

print("end ",b1)
a = b1[0:int(length / 2),0:int(weigth / 2)]
c = b1[0:int(length / 2), int(weigth / 2):weigth]
b = b1[int(length / 2):length, 0:int(weigth / 2)]
d = b1[int(length / 2):length, int(weigth / 2):weigth]
print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)

# print('end:', b1)
# print(b1[:,1])

# for i in b1:
#     print(i)
