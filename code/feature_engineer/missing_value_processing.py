"""
    author: Qinhsiu
    date: 2022/11/17
"""


import numpy as np
import pandas as pd
import copy


def create_data():
    df=pd.DataFrame({"A":list(range(1,11)),"B":list(range(11,21)),"C":list(range(101,111))})
    df.iloc[0,2]=np.nan
    df.iloc[1,2]=np.nan
    df.iloc[3,0]=np.nan
    df.iloc[4,2]=np.nan
    df.iloc[5,0]=np.nan
    df.iloc[6,2]=np.nan
    return df

# 模型预测
def model_predict(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error,r2_score
    # 查看特征相关性
    # print("特征之间相关性",df[["A","B","C"]].corr())

    # 使用模型进行预测
    train=df.copy().loc[~df["A"].isna(),["A","B"]].reset_index(drop=True)
    x=train[["B"]]
    y=train[["A"]]
    model=LinearRegression()
    model.fit(x,y)
    # print("r2 score: ",r2_score(y,x))
    # 使用预测值进行填充A
    df.loc[df["A"].isna(),["A"]]=model.predict(df.loc[df["A"].isna(),["B"]])

    train = df.copy().loc[~df["C"].isna(), ["B", "C"]].reset_index(drop=True)
    x = train[["B"]]
    y = train[["C"]]
    model_0=LinearRegression()
    model_0.fit(x,y)
    df.loc[df["C"].isna(), ["C"]] = model.predict(df.loc[df["C"].isna(), ["B"]])
    return df



# 常量或者统计数值进行填充
def fill_num(df,mode):
    """
    :param df: 需要填充的数据
    :param mode: 填充模式
    :return: 返回填充后的数据
    """
    from sklearn.impute import SimpleImputer,KNNImputer

    df_temp=copy.deepcopy(df)

    if mode==0:
        df_temp=df_temp.fillna(0)
    elif mode==1:
        # 填充均值
        df_mean=SimpleImputer(missing_values=np.nan,strategy="mean",copy=False)
        df_mean.fit_transform(df)
    elif mode==2:
        # 众数填充
        df_most_frequent=SimpleImputer(missing_values=np.nan,strategy="most_frequent",copy=False)
        df_most_frequent.fit_transform(df_temp)
    elif mode==3:
        # 中位数
        df_median=SimpleImputer(missing_values=np.nan,strategy="median",copy=False)
        df_median.fit_transform(df_temp)
    elif mode==4:
        # 先前项填充，然后后项
        df_temp.fillna(method="ffill",axis=0,inplace=True)
        df_temp.fillna(method="bfill", axis=0, inplace=True)
    elif mode==5:
        # 先后项填充，后前项填充
        df_temp.fillna(method="bfill",axis=0,inplace=True)
        df_temp.fillna(method="ffill", axis=0, inplace=True)
    elif mode==6:
        # KNN填充
        df_knn=KNNImputer(n_neighbors=3,copy=False)
        df_knn.fit_transform(df_temp)
    else:
        # 预测填充
        df_temp=model_predict(df_temp)

    return df_temp




if __name__ == '__main__':
    df=create_data()
    df_new=fill_num(df,0)
    print(df,df_new)




