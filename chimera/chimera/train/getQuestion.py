########1.读取train数据，获取输入question
import json
filename = "../../../../../data/ShareGPT_Vicuna_unfiltered/1280test.json"########训练数据
savefilename =  "../../../../../data/ShareGPT_Vicuna_unfiltered/1280question.json"
with open(filename, 'r') as f:
    train = json.load(f)
    
Q = {'question':[]}
len = len(train)
c = 0
for i in train:
    c=c+1
    if c%500 == 0 :print("当前进度{}".format(c/len))
    for j  in i['conversations']:
        if j['from'] =='human':
            
            Q['question'].append(j)
            break
        



########2.保存所有question

with open( savefilename, "w") as json_file:
    json.dump(Q, json_file)

