import numpy as np

classes = np.zeros(20)
with open('newsdataset/data/test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
with open('newsdataset/data/class.txt','r',encoding='utf-8') as f:
    clsnam = f.readlines()
for i in range(len(clsnam)):
    clsnam[i] = clsnam[i].strip('\n')
for line in lines:
    if line == '' or line == '\n':
        continue
    line = line.strip('\n')
    content, label = line.split('\t')
    label = int(label)
    classes[label]+=1

for i in range(len(classes)):
    print(clsnam[i]+':'+str(int(classes[i])))