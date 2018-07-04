import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv('../results/cnn-sep/eng-cnn-sep-active.csv')

strategy_list=['lc','rs']
experiment_list=['state=0/foldid=0','state=0/foldid=1','state=0/foldid=2',
'state=1/foldid=0','state=1/foldid=1','state=1/foldid=2',
'state=2/foldid=0','state=2/foldid=1','state=2/foldid=2',
'state=3/foldid=0','state=3/foldid=1','state=3/foldid=2',
'state=4/foldid=0','state=4/foldid=1','state=4/foldid=2',
'state=5/foldid=0','state=5/foldid=1','state=5/foldid=2',
'state=6/foldid=0','state=6/foldid=1','state=6/foldid=2',
'state=7/foldid=0','state=7/foldid=1','state=7/foldid=2',
'state=8/foldid=0','state=8/foldid=1','state=8/foldid=2',
'state=9/foldid=0','state=9/foldid=1','state=9/foldid=2',
                 ]

percentage_list=[0.2,0.3,0.5,0.6,0.75,0.9,1.0]
percentage_list2=[0,0.2,0.3,0.5,0.6,0.75,0.9,1.0]
number_list=[]

instance_list=[18,28,38,48,58,68,78,88,98,108,118,128,138,148,158,166]

def newEmptyRow(list,count):
   for i in range(0,count):
        row=[]
        list.append(row)
   return list

def list_average(list):
    list=np.array(list)
    return np.average(list)

def list_std(list,std):
    list = np.array(list)
    if std==0:
        return np.std(list,ddof = 1)
    else:
        return np.std(list)


times=[]
newEmptyRow(times,len(strategy_list))
for i in range(0,len(strategy_list)):
    newEmptyRow(times[i],30)

for index in df1.index:
    for i in range(0,len(strategy_list)):
        if df1.loc[index].values[3] == strategy_list[i]:
            experiment_no=df1.loc[index].values[1]
            experiment_index=experiment_list.index(experiment_no)
            times[i][experiment_index].append(df1.loc[index].values[5])

average=[]
std=[]

newEmptyRow(average,len(percentage_list))
newEmptyRow(std,len(percentage_list))

f1_average = []
newEmptyRow(f1_average, len(strategy_list))

for i in range(0,len(strategy_list)):
    print (strategy_list[i])

    f1=[]
    newEmptyRow(f1, len(percentage_list))
    for j in range(0,30):
        total=0
        for k in range(0,len(times[i][j])):
            percentage=instance_list[k]*1.0/166
            for m in range(0,len(percentage_list2)-1):
                if percentage>=percentage_list2[m] and percentage<percentage_list2[m+1]:
                    f1[m].append(times[i][j][k])

    for j in range(0,len(f1)):
        print ('average',list_average(f1[j]))
        print ('std',list_std(f1[j],1))
        average[j].append(list_average(f1[j]))
        std[j].append(list_std(f1[j],1))


    # preparation for paired t-test
    for j in range(0,len(instance_list)):
        total=0
        count=0
        for k in range(0,30):
            total+=times[i][k][j]
            count+=1
        f1_average[i].append(total / count)


# best and comparable

t_value=[31.821,4.541,3.143,2.896,2.764,2.650,2.602]

# significance level=0.98
def paired_tTest(strategy1,strategy2,percentage):
    x=[]
    index=0

    if percentage==len(percentage_list)-1:
        index=len(instance_list)
    else:
        for i in range(0,len(instance_list)):
            if instance_list[i]/166.0>percentage_list[percentage]:
                index=i
                break

    for i in range(0,index):
        x.append(abs(f1_average[strategy1][i]-f1_average[strategy2][i]))
    x_mean=list_average(x)
    x_std=list_std(x,0)
    n=len(x)
    t=x_mean/(x_std/(n**0.5))
    if t<t_value[percentage]:
        return 1
    else:
        return 0

for i in range(0,len(percentage_list)):
    print ('percentage',percentage_list[i])
    best=max(average[i])
    best_index=average[i].index(best)
    print ('best',strategy_list[best_index])
    for j in range(0,len(strategy_list)):
        if j!=best_index and paired_tTest(j,best_index,i)==1:
            print('comparable',strategy_list[j])


plt.plot(instance_list, f1_average[0], 'g', label='lc')
plt.plot(instance_list, f1_average[1], 'r', label='rs')


plt.xlabel('Number of Queries')
plt.ylabel('F1')
plt.title('Experiment Result')
plt.legend(loc='lower right')
plt.show()
