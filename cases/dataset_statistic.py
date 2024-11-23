import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import json
import ast

# 假设你的数据存储在一个名为 'dataset.txt' 的文本文件中
rebel_file_path1 = '../datasets/GPT3.5_result_rebel_sub/test.source'

# 读取文件并解析每行数据
with open(rebel_file_path1, 'r') as file:
    lines = file.readlines()
    dataset_rebel = [ast.literal_eval(line.strip()) for line in tqdm(lines)]

# 统计每个样本的三元组数量
triple_counts_rebel = [len(item) for sublist in dataset_rebel for item in sublist]

# 使用 Counter 来统计每个样本三元组数量的频率
frequencies_rebel = Counter(triple_counts_rebel)

# 计算总样本数量
total_samples_rebel = sum(frequencies_rebel.values())
#==================================================================================================
# 假设你的数据存储在一个名为 'dataset.txt' 的文本文件中
rebel_file_path1 = '../datasets/GPT3.5_result_GenWiki-Hard/test.source'

# 读取文件并解析每行数据
with open(rebel_file_path1, 'r') as file:
    lines = file.readlines()
    dataset_genwiki = [ast.literal_eval(line.strip()) for line in tqdm(lines)]

# 统计每个样本的三元组数量
triple_counts_genwiki = [len(item) for sublist in dataset_genwiki for item in sublist]

# 使用 Counter 来统计每个样本三元组数量的频率
frequencies_genwiki = Counter(triple_counts_genwiki)

# 计算总样本数量
total_samples_genwiki = sum(frequencies_genwiki.values())

#==================================================================================================
# 假设你的数据存储在一个名为 'dataset.txt' 的文本文件中
rebel_file_path1 = '../datasets/GPT3.5_result_SCIERC/test.source'

# 读取文件并解析每行数据
with open(rebel_file_path1, 'r') as file:
    lines = file.readlines()
    dataset_sci= [ast.literal_eval(line.strip()) for line in tqdm(lines)]

# 统计每个样本的三元组数量
triple_counts_sci = [len(item) for sublist in dataset_sci for item in sublist]

# 使用 Counter 来统计每个样本三元组数量的频率
frequencies_sci = Counter(triple_counts_sci)

# 计算总样本数量
total_samples_sci = sum(frequencies_sci.values())


print(f"rebel-sub: {total_samples_rebel}, genwiki: {total_samples_genwiki}, scierc :{total_samples_sci}")