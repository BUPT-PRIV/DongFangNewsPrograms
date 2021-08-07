import pickle
import numpy


f = open('newsdataset/data/test.txt','r')
f = f.readlines()
data = []
for line in f:
    zeros = numpy.zeros((2831),dtype=object)
    line = line.strip('\n')
    if line == '':
        continue
    text,a,b,c,d = line.split('\t')
    a,b,c,d = int(a),int(b),int(c),int(d)
    zeros[a] = 1
    zeros[b] = 1
    zeros[c] = 1
    zeros[d] = 1
    print(zeros[1000],zeros[a])
    one = (text,zeros)
    data.append(one)

f = open('/home/zhouyang/Bert-Multi-Label-Text-Classification-master/pybert/dataset/kaggle.valid.pkl', 'wb') #二进制打开，如果找不到该文件，则创建一个
pickle.dump(data, f) #写入文件


