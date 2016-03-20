# 1.data
> 用于存放数据集

1. mars_tianchi_songs.csv 初始歌曲数据
2. mars_tianchi_user_actions.csv 初始用户行为数据
3. mars_tianchi_user_actions_testing.csv 切割出的用于测试集的用户行为数据
4. mars_tianchi_user_actions_training.csv 切割出的用于训练集的用户行为数据
5. label.csv 从3中提取出的匹配提交要求的统计数据
6. mars_tianchi_songs_tiny.csv 歌曲数据的小数据
7. mars_tianchi_user_actions_tiny.csv 用户行为的小数据 
8. mars_tianchi_user_actions_tiny_testing.csv 用于测试集的用户行为的小数据
9. mars_tianchi_user_actions_tiny_training.csv 用于训练集的用户行为的小数据
10.label_tiny.csv 从8中提取出的统计数据

# 2.doc
> 用于存放文档，描述性文件

1. project_framework.md 用于描述整个项目框架。
2. interface.md 用于描述一些公共接口使用。


# 3.src
> 存放源码

1. logging.conf: python.logging的配置文件

## 3.1 script

> 放一些简单的脚本，用于数据调研类的

1. data_statistics.sh: 用于统计数据规模 (./data_statistics.sh)
2. data_split.py: 用于切割数据，生成相应的训练测试集，以及label数据 (./data_split.py <number of days for testing>)


## 3.2 utils

> 公共的接口，工具

