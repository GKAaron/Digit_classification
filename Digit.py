import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=150)
f1 = open("trainingimages",'r')
training_data = f1.read().splitlines()
f1.close()
f2 = open("testimages","r")
test_data = f2.read().splitlines()
f2.close()
f3 = open("traininglabels",'r')
training_label = list(map(int,f3.read().splitlines()))
f3.close()
f4 = open("testlabels",'r')
test_label = list(map(int,f4.read().splitlines()))
f4.close()
training_digit_class = np.zeros(10,dtype='float64')
digit_pro = dict()
for i in range(10):
    digit_pro[i] = np.zeros((28,28),dtype='float64')
for i in range(len(training_label)):
    d = list()
    digit = np.zeros((28, 28), dtype='float64')
    training_digit_class[training_label[i]] += 1
    for j in range(i*28,(i+1)*28):
        d.append(list(training_data[j]))
    data = np.array(d)
    digit[data!=' '] = 1
    digit_pro[training_label[i]] += digit
training_digit_class_pro = training_digit_class/len(training_label)
smoothing = np.zeros((28,28),dtype='float64')
k = 1
smoothing += k
for i in range(10):
    digit_pro[i] += smoothing
    digit_pro[i] /= (training_digit_class[i]+2*k)
training_digit_class_pro = np.log(training_digit_class_pro)
test_digit_class = np.zeros(10,dtype='float64')
confusion_matrix = np.zeros((10,10),dtype='float64')
max_posterior = np.array([None for x in range(10)])
min_posterior = np.array([None for x in range(10)])
max_posterior_example = np.empty(10,dtype='object')
min_posterior_example = np.empty(10,dtype='object')
pro_matrix = np.zeros((28, 28), dtype='float64')
a = np.zeros(10)
b = np.zeros(10)
for i in range(len(test_label)):
    test_digit_class[test_label[i]]+=1
    d = list()
    pro_max = None
    infer_class = None
    for j in range(28*i,28*(i+1)):
        d.append(list(test_data[j]))
    data = np.array(d)
    for t in range(10):
        pro_matrix[data!=' '] = digit_pro[t][data!=' ']
        pro_matrix[data==' ']= 1 - digit_pro[t][data==' ']
        pro_matrix = np.log(pro_matrix)
        pro = training_digit_class_pro[t] + pro_matrix.sum()
        if pro_max is None or pro > pro_max:
            pro_max = pro
            infer_class = t
        pro_matrix[:] = 0
    confusion_matrix[test_label[i]][infer_class] += 1
    if max_posterior[infer_class] is None or pro_max > max_posterior[infer_class]:
        max_posterior[infer_class] = pro_max
        max_posterior_example[infer_class] = d
        a[infer_class] = test_label[i]
    if min_posterior[infer_class] is None or pro_max < min_posterior[infer_class]:
        min_posterior[infer_class] = pro_max
        min_posterior_example[infer_class] = d
        b[infer_class] = test_label[i]
correct_classification = 0
for i in range(10):
    correct_classification += confusion_matrix[i][i]
    confusion_matrix[i]/=test_digit_class[i]
f5 = open('result','w')
print("The smoothing constant is",k,file=f5)
print("The overall accuracy is",correct_classification/len(test_label),file=f5)
print("The confusion matrix is:",file=f5)
print(confusion_matrix,file=f5)
for i in range(10):
    print('The highest posterior image of digit',i,":",file=f5)
    for j in max_posterior_example[i]:
        f5.writelines(''.join(j)+'\n')
for i in range(10):
    print('The lowest posterior image of digit', i, ":", file=f5)
    for j in min_posterior_example[i]:
        f5.writelines(''.join(j)+'\n')
f5.close()
print(correct_classification/len(test_label))
odd_matrix = np.array(confusion_matrix)
for i in range(10):
    odd_matrix[i][i] = np.NaN
odd_pair = np.zeros((4,2))
for i in range(4):
    index = np.unravel_index(np.nanargmax(odd_matrix),(10,10))
    odd_pair[i][0] = index[0]
    odd_pair[i][1] = index[1]
    odd_matrix[index[0]][index[1]] = np.NaN
fig = plt.figure()
cmap = plt.cm.get_cmap('rainbow')
for i in range(4):
    pro_first = np.log(digit_pro[odd_pair[i][0]])
    pro_second = np.log(digit_pro[odd_pair[i][1]])
    ax1 = fig.add_subplot(4,3,3*i+1)
    im1 = ax1.imshow(pro_first,cmap=cmap)
    plt.colorbar(im1)
    ax2 = fig.add_subplot(4,3,3*i+2)
    im2 = ax2.imshow(pro_second,cmap=cmap)
    plt.colorbar(im2)
    ax3 = fig.add_subplot(4,3,3*i+3)
    im3 = ax3.imshow(pro_first-pro_second,cmap=cmap)
    plt.colorbar(im3)
plt.show()






