"""
    author: Qinhsiu
    date: 2022/11/18
"""

# 数据处理
import pandas as pd
import numpy as np


# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid',font_scale=1.3)
plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False

# 特征工程
import sklearn
from sklearn import preprocessing                            #数据预处理模块
from sklearn.preprocessing import LabelEncoder               #编码转换
from sklearn.preprocessing import StandardScaler             #归一化
from sklearn.model_selection import StratifiedShuffleSplit   #分层抽样
from sklearn.model_selection import train_test_split         #数据分区
from sklearn.decomposition import PCA

# 分类算法
from sklearn.ensemble import RandomForestClassifier     #随机森林
from sklearn.svm import SVC,LinearSVC                   #支持向量机
from sklearn.linear_model import LogisticRegression     #逻辑回归
from sklearn.neighbors import KNeighborsClassifier      #KNN算法
from sklearn.cluster import KMeans                     #K-Means 聚类算法
from sklearn.naive_bayes import GaussianNB              #朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier         #决策树

# 集成学习
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 模型评估
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score  #分类报告
from sklearn.metrics import confusion_matrix           #混淆矩阵
from sklearn.metrics import silhouette_score           #轮廓系数（评价k-mean聚类效果）
from sklearn.model_selection import GridSearchCV       #交叉验证
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 用于缺失值填充
def model_predict(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error,r2_score
    # 查看特征相关性
    print("特征之间相关性",df[df.columns].corr())

    # 使用模型进行预测
    train=df.copy().loc[~df["Age"].isna(),["Age","Pclass","SibSp","Parch","Fare"]].reset_index(drop=True)
    x=train[["Pclass","SibSp","Parch","Fare"]]
    y=train[["Age"]]
    model=LinearRegression()
    model.fit(x,y)

    # # 使用预测值进行填充Age
    df.loc[df["Age"].isna(),["Age"]]=model.predict(df.loc[df["Age"].isna(),["Pclass","SibSp","Parch","Fare"]])

    return df




def get_data(data_path):
    """各个特征字段解释
    PassengerID： 乘客 ID
    Pclass： 舱位等级 (1 = 1st, 2 = 2nd, 3 = 3rd)
    Name： 乘客姓名
    Sex： 性别
    Age： 年龄
    SibSp： 在船上的兄弟姐妹／配偶个数
    Parch： 在船上的父母／小孩个数
    Ticket： 船票信息
    Fare： 票价
    Cabin： 客舱
    Embarked： 登船港口 (C = Cherbourg, Q = Queenstown, S = Southampton)
    survived:  变量预测为值 0 或 1（这里 1 表示幸存，0 表示遇难
    """
    df=pd.read_csv(data_path)
    # 查看数据
    print(df.head())
    print("Data shape:",df.shape)
    # 查看数据以及分布
    print(df.describe())

    # 统计缺失值
    print(df.isnull().sum())
    # 查看重复值
    print(df.duplicated().sum())

    # 查看数据类型
    print(df.info())

    # 查看数据统计
    print(df.count())

    # 删除一些缺失值较多的列
    df=df.drop(['Cabin'],axis=1)

    # 删除存在缺失数据的行
    # df=df.dropna(subset={"Age"},inplace=False)

    # 使用模型预测值填充年龄缺失值
    df=model_predict(df)

    # 使用众数填充港口缺失值
    # df.loc[df["Embarked"].isna(), ["Embarked"]]=
    # df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode())


    # 对分类数据进行独热编码
    return df




# 查看特征相关性
def corr(df):
    (df.corr().loc["Survived"].plot(kind="barh",figsize=(5,5)))
    plt.show()







if __name__ == '__main__':
    data_path="titanic.csv"
    df=get_data(data_path)
    model_predict(df)



