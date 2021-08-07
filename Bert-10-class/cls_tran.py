import os

cltran = [1,10,1,14,1,18,0,3,4,4]
with open('THUCNews/data/test.txt','r',encoding='utf-8') as f:
        more = f.readlines()
for i in range(len(more)):
    more[i] = more[i].strip('\n')
    if more[i] == '':
        more.remove('')
texts = []
cls = []
for i in range(len(more)):
    text,classes = more[i].split('\t')
    classes = str(cltran[int(classes)])
    texts.append(text)
    cls.append(classes)

after = ''
for a,b in zip(texts,cls):
    after += a
    after += '\t'
    after += b
    after += '\n' 

with open('enhance.txt','w',encoding='utf-8') as f:
    f.write(after)

