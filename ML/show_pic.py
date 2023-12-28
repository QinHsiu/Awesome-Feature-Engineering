import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
print(os.getcwd())

import json
with open("select_feat.txt","r+") as fr:
#     feature=yaml.load(fr)
    data=fr.readlines()
# data=json.loads(data)
    
# jiexi=lambda x:(map(float,x[0]),map(str,x[1]))
temp_dic={}
idx=0
temp=data[0].split(": [")
for idx in range(4):

    if idx==0:
        k=temp[0].split("{")[1]
        v=temp[1].split("]")[0].split("[")[0]
        k1=temp[1].split("]")[1].split(",")[1]

    if idx>=1 and idx<3: 
        k=k1
        v=temp[idx+1].split("]")[0].split("[")[0]
        k1=temp[idx+1].split("]")[1].split(",")[1]
    
    if idx==3:
        k=k1
        v=temp[-1].split("]")[0].split("[")[0]
    
    
    idx+=1
    
#     print(idx,k,type(v))
    temp_dic[k]=list(eval(v))
# print(temp_dic)

data=pd.DataFrame(temp_dic)
print(data)

keys=list(data.columns)
v=[kv[0] for kv in data.loc[:3,keys[0]].tolist()]
k=[kv[1] for kv in data.loc[:3,keys[0]].tolist()]
# data.columns

temp_dic={}
for key in data.columns:
    temp_data=data.loc[:,key].tolist()
    for v,k in temp_data:
        if k not in temp_dic:
            temp_dic[k]=[]
        temp_dic[k].append(v)

res_dic_sort=sorted(temp_dic.items(),key=lambda x: sum(x[1])/4)
res=res_dic_sort[:10]+res_dic_sort[-10:]
print(res)
res=list((kv[0],kv[1]) for kv in res)
plot_temp(res,"MergebyAverage")
    

def cnt_kv(df,c_name):
    v=[kv[0] for kv in df.loc[:,c_name].tolist()]
    k=[kv[1] for kv in df.loc[:,c_name].tolist()]
    temp_dic=dict(zip(k,v))
    sort_dic=sorted(temp_dic.items(),key=lambda x:x[1])
    top10=sort_dic[10][1]
    sorted_dic_r=sort_dic[:10]+sort_dic[-10:]
    k=[kv[0] for kv in sorted_dic_r]
    v=[kv[1] for kv in sorted_dic_r]
    new_dic=dict(zip(k,v))
    # 统计
    return new_dic

res_dic={}
for c_name in keys:
    temp_dic=cnt_kv(data,c_name)
    for k in temp_dic:
        if k not in res_dic:
            res_dic[k]=[]
        res_dic[k].append(temp_dic[k])
            
res_dic_sort=sorted(res_dic.items(),key=lambda x: (max(x[1]),min(x[1])))

res=res_dic_sort[:10]+res_dic_sort[-10:]

def plot_temp(temp_list,c_name):
    df=pd.DataFrame()
    df["k"]=[kv[0] for kv in temp_list]
    df["v"]=[sum(kv[1])/len(kv[1]) for kv in temp_list]
    
    top10=sorted(df["v"])[10]
    df['colors'] = ['red' if x < top10 else 'green' for x in df['v']]
    df.sort_values('v', inplace=True)
    df.reset_index(inplace=True)
    # params
    plt.rcParams['font.size'] = 20
    
    # Draw plot
    plt.figure(figsize=(30,10), dpi= 80)
    plt.hlines(y=df.index, xmin=0.6, xmax=df.v, color=df.colors, alpha=0.4, linewidth=5)

    # Decorations
    plt.gca().set(ylabel='$Feat$', xlabel='$Accuracy$')
    plt.yticks(df.index, df.k, fontsize=20)
    plt.title('Accuracy of top 10 and tail 10 features on method {}'.format(c_name), fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
#     plt.show()
    
    plt.savefig("./feat_exp/pics/{}.pdf".format(c_name))

plot_temp(res,"Merge")
    

# draw pic
def plot_hline(df,c_name):
#     x = df.loc[:, [c_name]]
    v=[kv[0] for kv in df.loc[:,c_name].tolist()]
    k=[kv[1] for kv in df.loc[:,c_name].tolist()]
    
    temp_dic=dict(zip(k,v))
    sort_dic=sorted(temp_dic.items(),key=lambda x:x[1])
    
    top10=sort_dic[10][1]
    
    sorted_dic_r=sort_dic[:10]+sort_dic[-10:]
    new_df=pd.DataFrame()
    new_df["k"]=[kv[0] for kv in sorted_dic_r]
    new_df["v"]=[kv[1] for kv in sorted_dic_r]
#     df["k"]=k[:10]+k[-10:]
#     df["v"]=v[:10]+v[-10:]
    df=new_df
#     print(df.columns)
    
    df['colors'] = ['red' if x < top10 else 'green' for x in df['v']]
    df.sort_values('v', inplace=True)
    df.reset_index(inplace=True)
    # params
    plt.rcParams['font.size'] = 20
    
    # Draw plot
    plt.figure(figsize=(30,10), dpi= 80)
    plt.hlines(y=df.index, xmin=0.0, xmax=df.v, color=df.colors, alpha=0.4, linewidth=5)

    # Decorations
    plt.gca().set(ylabel='$Feat$', xlabel='$Accuracy$')
    plt.yticks(df.index, df.k, fontsize=20)
    plt.title('Accuracy of top 10 and tail 10 features on method {}'.format(eval(c_name)), fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
#     plt.show()
    
    plt.savefig("./feat_exp/pics/{}.pdf".format(eval(c_name)))

plot_hline(data,keys[0])
