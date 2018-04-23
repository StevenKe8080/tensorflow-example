import pandas as pd
import tensorflow as tf

# 当前目录下的训练和测试文件
TRAIN_FILE = "iris_training.csv"
TEST_FILE = "iris_test.csv"

# 定义每列的属性和分类
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# 读取数据
def load_data(y_name='Species'):
    """返回(train_x, train_y), (test_x, test_y)这种样子的数据 """

    # 指定CSV_COLUMN_NAMES为列名读取csv文件
    train = pd.read_csv(TRAIN_FILE, names=CSV_COLUMN_NAMES, header=0)
    # train_x为前4列作为训练特征，train_y最后一列作为训练标签
    train_x, train_y = train, train.pop(y_name)

    # 测试数据处理同上
    test = pd.read_csv(TEST_FILE, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """训练输入函数"""
    # 输入值转换成数据集
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 打乱,重复,批处理数据
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # 返回数据
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels, batch_size):
    """一个评估和预测函数"""
    features=dict(features)
    if labels is None:
        # 没有标签就只用特征,无标签样本时使用
        inputs = features
    else:
        inputs = (features, labels)

    # 转成数据集.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # 批量处理文件
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # 返回数据集
    return dataset.make_one_shot_iterator().get_next()