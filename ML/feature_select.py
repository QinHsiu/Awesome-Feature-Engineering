# 使用机器学习方法来进行特征筛选
# 使用特征筛选的方法去除冗余特征，冗余特征是指可以由其他特征进行线性组合得到，并且对模型效果影响不大的特征
# 特征筛选使用的方案如下：
    # 使用传统的方法（皮尔逊相关系数、方差，最大信息系数等）
    # 使用递归特征删除方法
    # 使用学习方法
    # 使用集成学习方法（随机森林、集成袋子、投票、增强等）
# 参考资源[link](https://scikit-learn.org/stable/modules/ensemble.html)


# 导入相关包
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression,LinearRegression,RidgeCV,LassoCV
# 回归模型
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
# from sklearn.ensemble import VotingRegressor
# 分类模型
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
# from sklearn.ensemble import HistGradientBosstingClassifier
# from sklearn.ensemble import DecisionTreeClassifier
# bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# voting
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
# stacking
# from sklearn.ensemble import StackingRegressor
# adaboost
from sklearn.ensemble import AdaBoostClassifier
# tools
from itertools import product
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
# from sklearn.cross_validation import cross_val_score,ShuffleSplit
from sklearn.feature_selection import SelectKBest,chi2,VarianceThreshold,RFE,SelectFromModel
from minepy import MINE
import tensorflow as tf
import yaml







# 定义相关函数
def dataLoader(data_path,feature_config):
    idx=0
    num_epoch=1
    batch_size=1024
    totalData={}
    parse_fn=gen_thor_example_parse_fn(feature_config)
    dataset=input_fn_v2(data_path,batch_size,parse_fn,num_epoch)
    iterator = dataset.make_one_shot_iterator()  
    next_data = iterator.get_next()  
    try:
        while True:
            batchData=sess.run(next_data)
            for k in batchData:
                if k not in totalData:
                    totalData[k]=[]
                totalData[k].extend(batchData[k])
#             print("the length of a batch data is {}".format(len(batchData["purchase_7d"])))
            idx+=1
    except:
        print("end of batch, and batch nums is {}".format(idx))

    totalData=pd.DataFrame(totalData,index=range(len(totalData["purchase_7d"])))
    return totalData

    
def SelectFromEnsemble(modelName,data,target,task="clf",n_estimators=20, max_depth=4,min_samples_split=2,random_state=0):
    """
        modelName: model's name
        data: feature data
        target: label
        task: {"clf":claffication,"reg":regression}
        n_estimators: model nums
        max_depth: for tree model
        min_sample_split: for tree model
        random_state: seed
    """
    # 回归模型
    if modelName=="rfr":
        model=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    elif modelName=="gbr":
        model=GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1, max_depth=max_depth, random_state=random_state,loss='squared_error')   
    # 分类模型
    elif modelName=="rfc":
        model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,min_samples_split=min_samples_split, random_state=random_state)
    elif modelName=="dtc":
        model=DecisionTreeClassifier(n_estimators=n_estimators, max_depth=max_depth,min_samples_split=min_samples_split, random_state=random_state)
    elif modelName=="etc":
        model=ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,min_samples_split=min_samples_split, random_state=random_state)
    elif modelName=="gbc":
        model=GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0,max_depth=1, random_state=0)
    elif modelName=="hgbc":
        model=HistGradientBoostingClassifier(max_iter=100,min_samples_leaf=1,max_depth=max_depth,learning_rate=1)
    elif modelName=="bagging":
        model=BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    elif modelName=="voting":
        # classifier
        clf1 = LogisticRegression(random_state=1,solver='lbfgs')
        clf2 = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf3 = GaussianNB()
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
        # regressior
#         reg1 = GradientBoostingRegressor(random_state=1)
#         reg2 = RandomForestRegressor(random_state=1)
#         reg3 = LinearRegression()
#         ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])  
        feature_scores={}
        models=[clf1, clf2, clf3, eclf]
        model_names=['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']
        names=list(data.columns)
        scores=[]
        for idx, model_name_dic in enumerate(list(zip(models, model_names))):
            clf,label=model_name_dic
            score = cross_val_score(clf, data, target, scoring='accuracy', cv=5)
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (score.mean(), score.std(), label))
            feature_scores[model_names[idx]]=[]
            for i in range(data.shape[1]):
                #每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
                #ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
                score = cross_val_score(clf, data.iloc[:, i:i+1], target, scoring="accuracy",cv=ShuffleSplit(len(data), 3, .3))
                feature_scores[model_names[idx]].append((round(np.mean(score), 3), names[i]))
            feature_scores[model_names[idx]]=sorted(feature_scores[model_names[idx]],reverse=True)
        with open("res.txt","w+") as fw:
            fw.write("\n".join(["{}:{}".format(k,v) for k,v in feature_scores.items()]))
        return feature_scores
    elif mode=="stacking":
        estimators = [('ridge', RidgeCV()),
               ('lasso', LassoCV(random_state=42)),
               ('knr', KNeighborsRegressor(n_neighbors=20,
                                           metric='euclidean'))]
        final_estimator = GradientBoostingRegressor(n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,random_state=42)
        model = StackingRegressor(estimators=estimators,final_estimator=final_estimator)
    elif mode=="adaboost":
        model=AdaBoostClassifier(n_estimators=100)
    else:
        raise ValueError("The mode is error!")
#         raise NotImplementedError("No mode type, please retry.")
    # feature score
    scores=[]
    # feature names
    names=list(data.columns)
    for i in range(data.shape[1]):
        #每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
        #ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
        score = cross_val_score(model, data.iloc[:, i:i+1], target, scoring="r2",cv=ShuffleSplit(len(data), 3, .3))
        scores.append((round(np.mean(score), 3), names[i]))
        print(scores[-1])
    return sorted(scores,reverse=True)

# 使用传统的方法
def featureSelection(data,target,mode,k,modelName=None):
    # 使用皮尔逊相关系数，该方案只对线性相关性敏感，对非线性相关性无法判断
    if mode=="pearsonr":
#         print(list(map(lambda x:pearsonr(x, target), data.T)))
        selector=SelectKBest(lambda X, Y: np.array(list(map(lambda x:pearsonr(x, Y)[0], X.T))).T, k=k)
        select_features=pd.DataFrame(selector.fit_transform(data, target),columns=np.array(data.columns)[selector.get_support(indices=True)])
        return list(select_features.columns)[:5],list(select_features.columns)[-5:]
    # 使用卡方检验,该方案可以处理类别型变量，其考虑自变量与因变量样本频数的观察值与期望值的差距,需要数据是非负的
    elif mode=="kafang":
        # topk
        selector=SelectKBest(chi2, k=k).fit_transform(data, target)
        select_features=pd.DataFrame(selector.fit_transform(data, target),columns=np.array(data.columns)[selector.get_support(indices=True)])
        # tailk
        selector_t=SelectKBest(chi2, k=len(list(data.columns))-k).fit_transform(data, target)
        select_features_t=pd.DataFrame(selector_t.fit_transform(data, target),columns=np.array(data.columns)[selector_t.get_support(indices=True)])  
        return list(select_features.columns)[:5],set(list(data.columns)).difference(set(list(select_features_t.columns)))
    # 最大信息系数
    elif mode=="mine":
        def mic(x,y):
            m=MINE()
            m.compute_score(x,y)
            return (m.mic(),0.5)
        selector=SelectKBest(lambda X, Y: np.array(list(map(lambda x:mic(x, Y)[0], X.T))).T, k=k)
        select_features=pd.DataFrame(selector.fit_transform(data, target),columns=np.array(data.columns)[selector.get_support(indices=True)])
        # tailk
        selector_t=SelectKBest(lambda X, Y: np.array(list(map(lambda x:mic(x, Y)[0], X.T))).T, k=len(list(data.columns))-k)
        select_features_t=pd.DataFrame(selector_t.fit_transform(data, target),columns=np.array(data.columns)[selector_t.get_support(indices=True)])  
        return list(select_features.columns)[:5],set(list(data.columns)).difference(set(list(select_features_t.columns)))
    # 基于方差的方法，该方法仅仅适用于两个特征分布近乎一致的情况
    elif mode=="variance":
        return VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(data)
    # 使用递归特征消除法
    elif mode=="ref":
        if modelName=="lr":
            # 分类
            selector=RFE(estimator=LogisticRegression(penalty="l1", C=0.1), n_features_to_select=k)
            select_features=pd.DataFrame(selector.fit_transform(data, target),columns=np.array(data.columns)[selector.get_support(indices=True)])
            return list(select_features.columns)[:5],list(select_features.columns)[-5:]
    # 使用嵌入法，基于惩罚项的特征选择方法，通过正则项来选择特征，L1正则方法具有稀疏解，因此天然具备特征选择的特性
    elif mode=="embed":
        if modelName=="lr":
            selector=SelectFromModel(LogisticRegression(penalty="l1", C=0.1))
            select_features=pd.DataFrame(selector.fit_transform(data, target),columns=np.array(data.columns)[selector.get_support(indices=True)])
            return list(select_features.columns)[:5],list(select_features.columns)[-5:]
            # 也可以使用阈值
            # return SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(data, target)
        elif modelName=="liner":
            liner=LinearRegression().fit(data,target)
            selector=SelectFromModel(liner,prefit=True)
            select_features=pd.DataFrame(selector.transform(data))
            res_score=dict(zip(list(data.columns),liner.feature_importances_))
            scores=sorted(res_score.items(),key=lambda x:rescore[x],reverse=True)
            return scores[:5],scores[-5:]
            
            # re#turn list(select_features.columns)[:5]
    # 使用学习模型进行特征选择，直接使用目标模型建立预测模型
    elif mode=="learningmodel":
        return SelectFromEnsemble(modelName,data,target,n_estimators=10, max_depth=3)
    else:
        raise ValueError("The mode is error!")



# 进行特征打分选择
data_path=["part-r-01018.gz"]
feature_config_path = "cate1_prefer_v2.yml"
with open(feature_config_path, "r") as io:
    feature_config = yaml.load(io)
totalData=dataLoader(data_path,feature_config)
target=totalData["purchase_7d"]
# print("target: {}".format(target.head()))
data=totalData.drop(["purchase_7d","logid"],axis=1)
# print("features: {}".format(data.head()))
# print(data.info())

data=data.iloc[:1000,:]
target=target.iloc[:1000]

# mode="pearsonr"
# k=197
# select_features=featureSelection(data,target,mode,k,modelName=None)
# print("{} {}".format(mode,select_features))

res=""

mode="mine"
# k=len(data.columns)
k=5
# select_features=featureSelection(data,target,mode,k,modelName=None)
# print("{} {}".format(mode,select_features))
# res+="{} {}\n".format(mode,select_features)


mode="ref"
modelName="lr"
target=target.astype("int")
# select_features=featureSelection(data,target,mode,k,modelName=modelName)
# print("{} {}".format(mode,select_features))

# mode="embed"
# modelName="lr"
# select_features=featureSelection(data,target,mode,k,modelName=modelName)
# print("{} {}".format(mode,select_features))
# res+="{} {}\n".format(mode,select_features)

# mode="embed"
# modelName="liner"
# select_features=featureSelection(data,target,mode,k,modelName=modelName)
# print("{} {}".format(mode,select_features))
# res+="{} {}\n".format(mode,select_features)

# mode="learningmodel"
# modelName="rfr"
# select_features=featureSelection(data,target,mode,k,modelName=modelName)
# print("{} {}".format(mode,select_features))
# res+="{} {}\n".format(mode,select_features)


# mode="learningmodel"
# modelName="voting"
# select_features=featureSelection(data,target,mode,k,modelName=modelName)
# print("{} {}".format(mode,select_features))
# res+="{} {}\n".format(mode,select_features)

# with open("res.txt","w+") as fw:
#     fw.write(res)


mode="learningmodel"
modelName="voting"
select_features=featureSelection(data,target,mode,k,modelName=modelName)
print("{} {}".format(mode,select_features))
res+="{} {}\n".format(mode,select_features)





















