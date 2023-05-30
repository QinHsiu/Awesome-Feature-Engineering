# 推荐中常用的特征处理方法

- People generally consider DNNs as universal function approximators, that could potentially learn all kinds of feature interactions. However, recent studies found that DNNs are inefficient to even approximately model 2nd or 3rd-order feature crosses. -《DCN》作者

- 前记
    - 模型可以在一定程度上学习到特征的一些变换和组合，例如PNN、DCN、DeepFM、xDeepFM、AutoInt等模型可以建模特征的交叉组合，但是模型仍然很难学习到基于列的特征变换，这是因为模型一次只能接受一个批量大小的数据，无法建模全局的统计聚合信息，而这些信息往往是特别重要的
    - AutoFE自动特征组合主要依赖于特征变换、生成、搜索与评估，其无法自动识别场景的特殊性，难以评估特征子集的有效性
    - 好的特征应该具有的性质（区分性、特征之间相互独立、简单易于理解、伸缩性、高效性、灵活性、自适应）

- 用户侧
    - 用户粒度，也即从单个用户或者某一类用户出发，基于类别的统计对于缓解冷启动来说有重大意义
    - 时间粒度，从过去的几个小时，过去几周、过去几个月至今，这里在统计太长的时间粒度（例如首次使用至今）的时候需要考虑时间衰减，对于用户长期历史的统计（可以通过hadhoop/spark完成），对于用户短期历史的统计（可以直接访问线上redis缓存）
    - 物料粒度，可以是物品id，或者物品的属性，级别、标签等
    - 行为特征，包括正向（点击、有效观看时长、点赞、转发、收藏、评论等），负向（忽略、短播、踩、拉黑、不感兴趣等）
    - 统计对象包括次数、时长、金额等
    - 统计方法包括收集保存格式，计算XTR、计算占比等

- 物品侧
    - 对于item侧，最重要的应该是item的后验统计数据，主要有来自时间粒度和统计对象（CTR、平均播放进度、平均消费时长）的数据，这些统计数据是有偏的，也即有的商品推荐对了但是其不一定适合所有人，利用后验统计数据做特征会在一定程度上加剧马太效应，前期后验数据好的item可能会排得更加靠前，这样不利于新item的冷启动，这里可以另外建立一个模型根据现有的信息来预测后验数据

- 交叉特征
    - 直接对两两特征做特征交叉会导致特征数量暴涨，耗费大量的资源，另外其扩展性也特别差
    - 尝试的解决思路[《Explicit Semantic Cross Feature Learning via Pre-trained Graph Neural Networks for CTR Prediction》SIGIR2021](https://arxiv.org/pdf/2105.07752.pdf),该论文指出通过链接预估训练一个GNN模型，然后将用户侧与物品侧特征输入GNN，用输出的xtr作为特征，该方法节省了存储和计算开销，其基于embedding的，因此扩展性也比较好，借鉴上述思想可以将GNN换成FM模型来学习xtr，针对每一对特征点积之后再经过sigmoid函数，得到的xtr用于训练或者预测
- 特征收缩
    - 在计算梯度的时候数量级比较大的值对梯度的支配会比较重要，因此对具有不同的数量级的特征进行特征缩放处理是一个必要的操作，不做特征缩放，取值范围比较大的特征维度会支配距离函数的计算，使得其他特征失去原本应该有的作用，常用的特征缩放技术如下
    $$x_{norm}=\frac{x-\min(x)}{\max{x}-\min{x}} \in [0,1], Min-Max$$
    $$x_{norm}=\frac{x-mean(x)}{\max{x}-\min{x}} \in [0,1], Scale$$
    $$x_{norm}=\frac{x-mean(x)}{std(x)} \tilde N(0,1), Z-score$$
    $$x_{\log}=$\log(1+x), x_{\log-norm}=\frac{x_{\log}-mean(x_{\log})}{std(x_{\log})}, log-based$$
    $$x_{norm}=\frac{x}{||x||_{2}}, L2 normalization$$


- 技巧
    - 可以利用用户给物品反向打标签来缓解冷启动问题，例如top10好看的电影，这些都是用户消费反向给物品打上的极其重要的标签


- 参考资料
    - [推荐系统|特征工程中的技巧](https://mp.weixin.qq.com/s/SBeN0KKVJEroyzIsto04ig)

