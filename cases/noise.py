import torch
from transformers import BertTokenizer, BertModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(torch.device("cuda:0"))

# 定义一个函数，用于将单词列表分割成每5个一组的列表
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# 定义函数来计算文本嵌入
def get_sentence_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(torch.device("cuda:0"))
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embedding = outputs.last_hidden_state[:, 0, :]
    return sentence_embedding

# 给定的三元组数据
triples =[['Mikhailovsky Palace', 'located in the administrative territorial entity', 'Saint Petersburg'], ['Mikhailovsky Palace', 'country', 'Russia'], ['Mikhailovsky Palace', 'architectural style', 'neoclassicism'], ['Mikhailovsky Palace', 'architect', 'Carlo Rossi'], ['Saint Petersburg', 'country', 'Russia'], ['Russia', 'contains administrative territorial entity', 'Saint Petersburg'], ['Grand Duke Michael Pavlovich', 'father', 'Emperor Paul I'], ['Grand Duke Michael Pavlovich', 'sibling', 'Alexander I'], ['Emperor Paul I', 'child', 'Grand Duke Michael Pavlovich'], ['Alexander I', 'sibling', 'Grand Duke Michael Pavlovich'], ['Alexander I', 'father', 'Emperor Paul I']]


# 给定的文本
text = """
The Mikhailovsky Palace () is a grand ducal palace in Saint Petersburg, Russia. It is located on Arts Square and is an example of Empire style neoclassicism. The palace currently houses the main building of the Russian Museum and displays its collections of early, folk, eighteenth, and nineteenth century art. It was originally planned as the residence of Grand Duke Michael Pavlovich, the youngest son of Emperor Paul I. Work had not yet begun on the Mikhailovsky Palace, when Paul was overthrown and killed in a palace coup that brought Michael's elder brother to the throne as Alexander I. The new emperor resurrected the idea for a new palace by the time Michael was 22, and plans were drawn up by Carlo Rossi to develop a new site in Saint Petersburg. The palace, built in the neoclassic style, became the centrepiece of an ensemble that took in new streets and squares. It was lavishly decorated, with the interiors costing more than the main construction work. It was gifted to Grand Duke Michael and his new wife, Grand Duchess Elena Pavlovna, by the Emperor in 1825. The grand ducal family had comfortable apartments furnished to their individual tastes. Grand Duke Michael carried out some of his military duties there, while his wife hosted salons that brought together many of the leading members of Saint Petersburg society and culture. The Grand Duchess continued this lifestyle after her husband's death in 1849, until her own death in 1873. The palace was passed on to the couple's daughter, Grand Duchess Catherine Mikhailovna.Over the years of their residency, the family renovated and refurbished the palace's rooms in keeping with contemporary tastes. By the time of Grand Duchess Catherine's death in 1894, the staterooms were no longer in regular use—the family resided for the most part in the palace's wings. With the death of the Grand Duchess, the palace was inherited by her children, who were members of the family of the Dukes of Mecklenburg-Strelitz. Concerned about the palace passing out of the Romanov family, Emperor Alexander III decided to buy it back for the state. He died before this could be arranged, but the negotiations were carried out on behalf of his son Emperor Nicholas II, by Minister of Finance Sergei Witte. Nicholas gave it to the newly established Russian Museum, in honour of his father, with the remit that it collect and display domestic art. The palace was extensively renovated to fit its new role, with some of the interiors retained. One wing was demolished and rebuilt, later becoming the Russian Museum of Ethnography, while a new extension, the Benois wing, was added in the 1910s.
"""
refined_text = """
The Mikhailovsky Palace is a grand ducal palace in Saint Petersburg, Russia, located on Arts Square. It is an example of Empire style neoclassicism and currently houses the main building of the Russian Museum. The palace was originally planned as the residence of Grand Duke Michael Pavlovich, but construction began after Emperor Paul I was overthrown. The palace was gifted to Grand Duke Michael and his wife, Grand Duchess Elena Pavlovna, in 1825. Concerned about the palace leaving the Romanov family, Emperor Alexander III decided to buy it back for the state. The palace was then given to the Russian Museum by Emperor Nicholas II, with renovations to fit its new role.
"""


num_chunks = 20
words = text.split(" ")
chunk_size = len(words) // num_chunks
sentences = list(chunk_list(words, chunk_size))
sentences_original = [" ".join(x) for x in sentences][:num_chunks]

# 计算每个句子的BERT嵌入
print("Chunk Embedding...")
sentence_embeddings = [get_sentence_embedding(sentence, tokenizer, model) for sentence in tqdm(sentences_original)]

# 计算每个三元组的BERT嵌入
print("Triple Embedding...")
triple_embeddings = [get_sentence_embedding(' '.join(triple), tokenizer, model) for triple in tqdm(triples)]

# 计算余弦相似度
print("Similarity Matrix...")
cosine_similarities_original = torch.zeros((len(sentences_original), len(triples)))
i = 0
for sentence_emb in tqdm(sentence_embeddings):
    for j, triple_emb in enumerate(triple_embeddings):
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(sentence_emb, triple_emb)
        cosine_similarities_original[i, j] = cos_sim
    i += 1

# 将相似度矩阵转换为numpy数组
similarity_matrix_original = cosine_similarities_original.detach().numpy()

# refined
num_chunks = 20
words = refined_text.split(" ")
chunk_size = len(words) // num_chunks
sentences = list(chunk_list(words, chunk_size))
sentences_refined = [" ".join(x) for x in sentences][:num_chunks]

# 计算每个句子的BERT嵌入
print("Chunk Embedding...")
sentence_embeddings = [get_sentence_embedding(sentence, tokenizer, model) for sentence in tqdm(sentences_refined)]

# 计算余弦相似度
print("Similarity Matrix...")
cosine_similarities_refined = torch.zeros((len(sentences_refined), len(triples)))
i = 0
for sentence_emb in tqdm(sentence_embeddings):
    for j, triple_emb in enumerate(triple_embeddings):
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(sentence_emb, triple_emb)
        cosine_similarities_refined[i, j] = cos_sim
    i += 1

# 将相似度矩阵转换为numpy数组
similarity_matrix_refined = cosine_similarities_refined.detach().numpy()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)


# 创建第二个heatmap
sns.heatmap(similarity_matrix_refined, annot=False, cmap='Purples', ax=ax2,\
    xticklabels=[f'Triple {i+1}' for i in range(len(triples))], yticklabels=[f'Chunk {i+1}' for i in range(len(sentences_refined))])
ax2.tick_params(axis='y', labelsize=9)
ax2.tick_params(axis='x', labelsize=9)
# 获取第一个heatmap的颜色条范围
cbar = ax2.collections[0].colorbar
vmin, vmax = cbar.vmin, cbar.vmax

# 创建第一个heatmap
sns.heatmap(similarity_matrix_original, annot=False, cmap='Purples', ax=ax1,\
            xticklabels=[f'Triple {i+1}' for i in range(len(triples))], yticklabels=[f'Chunk {i+1}' for i in range(len(sentences_original))],
            vmax=vmax, vmin=vmin)
ax1.tick_params(axis='y', labelsize=9)
ax1.tick_params(axis='x', labelsize=9)

# 调整子图参数，以便为标签留出空间
plt.subplots_adjust(bottom=0.2, left=0.2)  # 注意调整right参数以适应两个子图

# 保存图像
plt.savefig(f"relavance_test185_comparison.jpg")
plt.show()