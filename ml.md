# 基于机器学习方法的特征工程

- 基于过滤的方法
    - 针对单一变量特征可以直接根据缺失、异常百分比、方差、频数等特征选择剔除或者保留该特征
    - 针对多变量特征而言主要考虑自变量与自变量（是否存在多重共线性等）、自变量与因变量（是否相关等）
    - 使用一些阈值或者经验来剔除一些不太重要的特征，该方法计算开销小，且有利于避免模型过拟合，缺点是没有考虑后续需要使用的学习器去选择特征子集，减弱学习器拟合能力
    - 删除未使用的特征列（最简单的方法是使用直接，凭借经验删除一些与目标不太相关的特征列）
    - 删除具有缺失值的列（当某一些特征列中缺失值较多且难以进行有效填充的情况下可以考虑直接进行删除）
    - 删除具有异常值的列（当某一些特征中的值明显处于离群点的时候可以考虑删除该特征，或者进行修正）
    - 删除与目标不太相关的列，利用相关性系数等方法进行区分，对于数值型特征可以直接使用相关性系数，对于分类变量可以使用箱线图来判别目标与分类特征之间的相关性
    $$p(X,Y)=\frac{cov(X,Y)}{\sigma_{X}\sigma_{Y}},皮尔逊相关系数$$
    $$p=1-\frac {6\sum^{n}_{i=1}{d_{i}^{2}}}{n(n^{2}-1)},斯皮尔曼相关系数$$
    其中皮尔逊相关系数是两个变量的协方差除以两个变量的标准差的乘积，其能够反映两个随机变量的相关程度，正相关与负相关，其中0表示不相关；斯皮尔曼相关系数是建立在变量符合正态分布的条件上的，如果变量是顺序变量可以采用该方法计算相关性，其中d表示两个变量的等级差值
    - 删除一些低方差的特征，需要注意的是这些特征的数量级，当数量级相差较大的时候，他们的方差也会相差特别大
        ```
        from sklearn.feature_selection import VarianceThreshold
        variance = VarianceThreshold(threshold = (.9 * (1 - .9)))
        variance.fit(data)
        variance.get_support()
        ```
    - 对于连续型和类别型特征可以采用方差分析ANOVA和计算肯德尔等级相关系数来计算两个特征之间的相关性
    - 对于类别型和类别型特征可以采用卡方检验和互信息方法来进行过滤
    - 根据多重共现性(一个或者多个变量依赖于另外一个变量，可能导致多重共线性)删除一些共线性比较大的特征，因为机器学习中假设不同的特征之间应该是独立于其他特征的，对于数值型数据可以使用热力图来检查和寻找相关特征，也可以使用方差膨胀因子的方法来确定多重共线性并删除相关特征，对于分类变量可以采用独立性卡方检验之类的统计检验方法
        ```
        from pandas_profiling import ProfileReport
        profile = ProfileReport (data, title = 'Loans Defaults Prediction', html = {'style': {'full_width': True }})
        profile
        ```
    - 特征系数，特征适应度的一个关键指标是回归系数（beta系数），它显示了模型中特征的相对贡献，可以删除贡献很小或者没有贡献的特征
    - p值，在回归中，p值用于显示预测变量与目标之间的关系是否具有统计显著性，如果一些特征不显著，可以将这些特征进行删除处理
    - 主成分分析，降低高维度特征空间的维度，原始特征会被重新映射到新的维度，最终目标是找到能解释数据方差的特征数量

- 基于包装的方法
    - 根据ML的训练结果来选择特征，每一个子集训练都得到一个分数，根据该分数添加或者删除特征，并在达到某个阈值的时候停止操作；该方法优点是特征选择更加具有针对性，对模型性能有好处，缺点是计算开销大，主要包含三类搜索算法（完全搜索、启发式搜索、随机搜索）
    - 完全搜索也即将所有可能的特征子集输入模型，直接根据最佳模型得分进行判断，该方法计算开销大
    - 基于启发式搜索的算法，利用模型得分或者特征权重不断缩小搜索空间，其可以分为前向/后向搜索、递归特征删除等
    - 随机搜索，随机选择多个特征子集，然后分别评估模型表现，最后根据评分选择合适的特征子集，比较出名的有Null Importantance方法
    - 基于特征重要性的选择，决策树/随机森林使用一个特征来分割数据，该特征最大程度地减少了噪音（使用基尼指数或者信息增益衡量），可以通过该方法来计算不同特征对于目标变量的重要性，然后据此删除一部分不太重要的特征
    - 使用sklearn库中的自动特征选择，基于卡方的技术、基于正则化、基于序贯法、递归特征删除等
        ```
        lr = LinearRegression(normalize=True)
        lr.fit(X,y)
        # 使用RFE的再次训练
        rfe = RFE(lr, n_features_to_select=1,verbose=3)
        rfe.fit(X,y)
        ranks["RFE"] = ranking(list(map(float, rfe.ranking_)),col_names,order=-1)
        ranks  # 特征和得分
        ```
        ```
        # 1、线性回归
        lr = LinearRegression(normalize=True)
        lr.fit(X,y)
        ranks["LinReg"] = ranking(np.abs(lr.coef_), col_names)
        # 2、岭回归
        ridge = Ridge(alpha=7)
        ridge.fit(X,y)
        ranks["Ridge"] = ranking(np.abs(ridge.coef_), col_names)
        # 3、Lasso回归
        lasso = Lasso(alpha=0.05)
        lasso.fit(X,y)
        ranks["Lasso"] = ranking(np.abs(lasso.coef_), col_names)
        ```

- 基于嵌入的方法
    - 结合前面两种方法，例如使用LASSO和树模型，其特征选择与模型训练过程有明显的区分，其优点是比包装法更加省时省力，把特征选择交给模型去学习，其缺点是增加模型训练负担
        ```
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier

        iris = load_iris()

        # 将待L1惩罚项的逻辑回归作为基模型的特征选择
        selected_data_lr = SelectFromModel(LogisticRegression(penalty='l1', C = 0.1, solver = 'liblinear'), max_features = 3).fit_transform(iris.data, iris.target)

        # 将GBDT作为基模型的特征选择
        selected_data_gbdt = SelectFromModel(GradientBoostingClassifier(), max_features = 3).fit_transform(iris.data, iris.target)

        print(iris.data.shape)
        print(selected_data_lr.shape)
        print(selected_data_gbdt.shape)
        ```

- 参考资料
    - [机器学习+过滤法](https://github.com/mabalam/feature_selection)
    - [四种特征选择方法比较](https://mp.weixin.qq.com/s/xjWOX-ZePXpmexcdISZ49Q)
    - [特征选择常用机器学习方法](https://mp.weixin.qq.com/s/xuFPLiPA9nzBPvcXAt5QRg)
    - [特征选择方法总结](https://mp.weixin.qq.com/s/PCShJQwDotCwsgAYKDd_aA)