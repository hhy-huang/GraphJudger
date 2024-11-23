import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import json
import ast

# 假设你的数据存储在一个名为 'dataset.txt' 的文本文件中
rebel_file_path1 = '../datasets/GPT3.5_result_rebel_sub/train.source'
rebel_file_path2 = '../datasets/GPT3.5_result_rebel_sub/test.source'

# 读取文件并解析每行数据
with open(rebel_file_path1, 'r') as file:
    lines = file.readlines()
    dataset_rebel = [[ast.literal_eval(line.strip())] for line in tqdm(lines)]
# 读取文件并解析每行数据
with open(rebel_file_path2, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        dataset_rebel.append(ast.literal_eval(line.strip()))


# 统计每个样本的三元组数量
triple_counts_rebel = [len(item) for sublist in dataset_rebel for item in sublist]

# 使用 Counter 来统计每个样本三元组数量的频率
frequencies_rebel = Counter(triple_counts_rebel)

# 计算总样本数量
total_samples_rebel = sum(frequencies_rebel.values())

# 计算每个样本三元组数量的占比
proportions_rebel = {k: v / total_samples_rebel for k, v in frequencies_rebel.items()}

# 准备绘图数据
labels_rebel = list(proportions_rebel.keys())
values_rebel = list(proportions_rebel.values())

# 对labels和values进行排序，以确保折线图的顺序正确
sorted_indices_rebel = sorted(range(len(labels_rebel)), key=labels_rebel.__getitem__)
labels_rebel = [labels_rebel[i] for i in sorted_indices_rebel]
values_rebel = [values_rebel[i] for i in sorted_indices_rebel]

#==================================================================================================
# 假设你的数据存储在一个名为 'dataset.txt' 的文本文件中
rebel_file_path1 = '../datasets/GPT3.5_result_GenWiki-Hard/train.source'
rebel_file_path2 = '../datasets/GPT3.5_result_GenWiki-Hard/test.source'

# 读取文件并解析每行数据
with open(rebel_file_path1, 'r') as file:
    lines = file.readlines()
    dataset_genwiki = [[ast.literal_eval(line.strip())] for line in tqdm(lines)]
# 读取文件并解析每行数据
with open(rebel_file_path2, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        dataset_genwiki.append(ast.literal_eval(line.strip()))


# 统计每个样本的三元组数量
triple_counts_genwiki = [len(item) for sublist in dataset_genwiki for item in sublist]

# 使用 Counter 来统计每个样本三元组数量的频率
frequencies_genwiki = Counter(triple_counts_genwiki)

# 计算总样本数量
total_samples_genwiki = sum(frequencies_genwiki.values())

# 计算每个样本三元组数量的占比
proportions_genwiki = {k: v / total_samples_genwiki for k, v in frequencies_genwiki.items()}

# 准备绘图数据
labels_genwiki = list(proportions_genwiki.keys())
values_genwiki = list(proportions_genwiki.values())

# 对labels和values进行排序，以确保折线图的顺序正确
sorted_indices_genwiki = sorted(range(len(labels_genwiki)), key=labels_genwiki.__getitem__)
labels_genwiki = [labels_genwiki[i] for i in sorted_indices_genwiki]
values_genwiki = [values_genwiki[i] for i in sorted_indices_genwiki]



#==================================================================================================
# 假设你的数据存储在一个名为 'dataset.txt' 的文本文件中
rebel_file_path1 = '../datasets/GPT3.5_result_SCIERC/train.source'
rebel_file_path2 = '../datasets/GPT3.5_result_SCIERC/test.source'

# 读取文件并解析每行数据
with open(rebel_file_path1, 'r') as file:
    lines = file.readlines()
    dataset_sci= [[ast.literal_eval(line.strip())] for line in tqdm(lines)]
# 读取文件并解析每行数据
with open(rebel_file_path2, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        dataset_sci.append(ast.literal_eval(line.strip()))


# 统计每个样本的三元组数量
triple_counts_sci = [len(item) for sublist in dataset_sci for item in sublist]

# 使用 Counter 来统计每个样本三元组数量的频率
frequencies_sci = Counter(triple_counts_sci)

# 计算总样本数量
total_samples_sci = sum(frequencies_sci.values())

# 计算每个样本三元组数量的占比
proportions_sci = {k: v / total_samples_sci for k, v in frequencies_sci.items()}

# 准备绘图数据
labels_sci = list(proportions_sci.keys())
values_sci = list(proportions_sci.values())

# 对labels和values进行排序，以确保折线图的顺序正确
sorted_indices_sci = sorted(range(len(labels_sci)), key=labels_sci.__getitem__)
labels_sci = [labels_sci[i] for i in sorted_indices_sci]
values_sci = [values_sci[i] for i in sorted_indices_sci]


# 使用 matplotlib 绘制折线图
values_rebel[10] += sum(values_rebel[11:])  # 将所有大于10的值加到第10个元素上
values_rebel = values_rebel[:11]  # 保留前11个元素，去掉多余的
labels_rebel = labels_rebel[:11]
values_genwiki[10] += sum(values_genwiki[11:])  # 将所有大于10的值加到第10个元素上
values_genwiki = values_genwiki[:11]  # 保留前11个元素，去掉多余的
labels_genwiki = labels_genwiki[:11]
values_sci[10] += sum(values_sci[11:])  # 将所有大于10的值加到第10个元素上
values_sci = values_sci[:11]  # 保留前11个元素，去掉多余的
labels_sci = labels_sci[:11]
plt.plot(labels_rebel, values_rebel, marker='o', label="REBEL-Sub")
plt.fill_between(labels_rebel, values_rebel, alpha=0.2)
plt.plot(labels_genwiki, values_genwiki, marker='o', label="GenWiKi")
plt.fill_between(labels_genwiki, values_genwiki, alpha=0.2)
plt.plot(labels_sci, values_sci, marker='o', label="SCIERC")
plt.fill_between(labels_sci, values_sci, alpha=0.2)

plt.xlabel('# of Triples')
plt.ylabel('Ratio')
plt.xticks(labels_rebel, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '>10'])
# 添加图例
plt.legend()
plt.grid(True)
plt.savefig('./dataset.jpg')
