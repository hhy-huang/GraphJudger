import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# 定义一个函数来计算文本长度区间
def get_length_intervals(lengths, interval=10, max_interval=300):
    intervals = [
        (min(length // interval, max_interval // interval) * interval)
        for length in lengths
    ]
    return intervals

# 读取和处理数据集的函数
def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    dataset = [line.strip() for line in tqdm(lines)]
    return dataset

# 统计文本长度并计算区间
def calculate_proportions(dataset, interval=10, max_interval=300):
    text_lengths = [len(item.split(" ")) for item in dataset]
    length_intervals = get_length_intervals(text_lengths, interval, max_interval)
    frequencies = Counter(length_intervals)
    total_samples = sum(frequencies.values())
    proportions = {k: v / total_samples for k, v in frequencies.items()}
    return proportions

# 处理不同数据集
rebel_file_paths = [
    '../datasets/GPT3.5_result_rebel_sub/train.target',
    '../datasets/GPT3.5_result_rebel_sub/test.target'
]
genwiki_file_paths = [
    '../datasets/GPT3.5_result_GenWiki-Hard/train.target',
    '../datasets/GPT3.5_result_GenWiki-Hard/test.target'
]
sci_file_paths = [
    '../datasets/GPT3.5_result_SCIERC/train.target',
    '../datasets/GPT3.5_result_SCIERC/test.target'
]

# 计算比例
proportions_rebel = calculate_proportions(
    [item for file_path in rebel_file_paths for item in process_file(file_path)]
)
proportions_genwiki = calculate_proportions(
    [item for file_path in genwiki_file_paths for item in process_file(file_path)]
)
proportions_sci = calculate_proportions(
    [item for file_path in sci_file_paths for item in process_file(file_path)]
)

# 准备绘图数据
def prepare_plot_data(proportions):
    labels = list(proportions.keys())
    values = list(proportions.values())
    sorted_indices = sorted(range(len(labels)), key=labels.__getitem__)
    labels = [labels[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    return labels, values

labels_rebel, values_rebel = prepare_plot_data(proportions_rebel)
labels_genwiki, values_genwiki = prepare_plot_data(proportions_genwiki)
labels_sci, values_sci = prepare_plot_data(proportions_sci)

# 使用 matplotlib 绘制折线图
plt.plot(labels_rebel, values_rebel, marker='o', label="REBEL-Sub")
plt.fill_between(labels_rebel, values_rebel, alpha=0.2)
plt.plot(labels_genwiki, values_genwiki, marker='o', label="GenWiKi")
plt.fill_between(labels_genwiki, values_genwiki, alpha=0.2)
plt.plot(labels_sci, values_sci, marker='o', label="SCIERC")
plt.fill_between(labels_sci, values_sci, alpha=0.2)

plt.xlabel('Length of Interval')
plt.ylabel('Ratio')
plt.xticks(labels_rebel, [f'{i}-{i+10}' if i < 3000 else '>300' for i in labels_rebel], rotation=45, fontsize=7)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./text_length_distribution.jpg')