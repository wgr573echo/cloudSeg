## 天绘一号新整理的训练过程
# 文件夹用处
- data：数据集
- experiments：配置文件
- lib：调用的函数：包括模型文件(seg_hrnet)、数据类(remote)
- log：日志文件
- output：测试结果输出
- pretrained_models：预训练模型
# 文件关联
- lib\models\seg_hrnet.py：建立模型实例、初始化参数，会复制lib\models到\hr\output\remote\seg_hrnet_forest\models
- lib\datasets\remote.py：构造Remote类，包含创建train_dataset迭代函数
- lib\core\criterion.py：损失函数
- lib\utils\utils.py：FullModel类，得到model实例 可以得到outputs和label
- lib\core\function.py：train、test、testval的函数
- lib\datasets\base_dataset.py：BaseDataset类，图像迭代之前的预处理、标准化参数之类的，以及送入模型最后一步
- D:\wgr\hr\output\remote\seg_hrnet_forest：模型保存的路径