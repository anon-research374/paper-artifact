import argparse
import os
import warnings
import nltk
from datasets import load_from_disk
import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm
from kmeans_pytorch import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 配置nltk数据路径
nltk_data_path = '/home/zhai/liyun/SemStamp/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
# 下载punkt
success = nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
if not success:
    print(f"请手动执行：\nmkdir -p {nltk_data_path} && python -m nltk.downloader -d {nltk_data_path} punkt")
    exit(1)
# 配置nltk路径
nltk.data.path.insert(0, nltk_data_path)
# 关闭警告
warnings.filterwarnings("ignore")


########################1. 计算embedding########################
# 多进程嵌入生成
def worker(rank, text_chunk, embedder_path, queue, encode_batch_size):
    """
    Worker function to process a text chunk and generate embeddings on a specific GPU.
    """
    device = f'cuda:{rank}'
    embedder = SentenceTransformer(embedder_path, device=device)
    embedder = embedder.eval()

    sent_embeds = []

    # Progress bar for each worker
    with tqdm(total=len(text_chunk), desc=f"Worker {rank} Encoding", position=rank) as pbar:
        for i in range(0, len(text_chunk), encode_batch_size):
            batch_texts = text_chunk[i:i + encode_batch_size]
            batch_embeds = embedder.encode(batch_texts, convert_to_tensor=True)
            # sent_embeds.extend(batch_embeds)
            sent_embeds.extend([e.cpu().numpy() for e in batch_embeds])
            pbar.update(len(batch_texts))

    # Put all embeddings into the queue
    queue.put(sent_embeds)


# 将数据集中的文本转换为 SentenceTransformer 生成的向量嵌入，并以 pickle 格式存储
def embed_gen_list(dataset_path, embedder_path, encode_batch_size=32, num_gpus=torch.cuda.device_count()):
    """
    Parallelized embedding generation for the dataset with progress bars.
    """
    from multiprocessing import Process, Queue

    dataset = load_from_disk(dataset_path)
    texts = dataset['text']
    flattened_texts = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        flattened_texts.extend(sentences)
    # Total progress bar
    total_progress = tqdm(total=len(flattened_texts), desc="Total Progress", position=num_gpus)
    # Split the dataset into chunks for each GPU
    text_chunks = [flattened_texts[i::num_gpus] for i in range(num_gpus)]
    # Queue to collect embeddings from workers
    queue = Queue()

    processes = []
    for rank, text_chunk in enumerate(text_chunks):
        p = Process(target=worker, args=(rank, text_chunk, embedder_path, queue, encode_batch_size))
        p.start()
        processes.append(p)

    # Collect embeddings from workers
    all_embeds = []
    while any(p.is_alive() for p in processes) or not queue.empty():
        while not queue.empty():
            chunk_embeds = queue.get()
            # all_embeds.extend(chunk_embeds)
            all_embeds.extend([torch.tensor(e).to('cuda') for e in chunk_embeds])
            total_progress.update(len(chunk_embeds))

    total_progress.close()

    for p in processes:
        p.join()

    # Save embeddings to a single pickle file
    name = os.path.join(dataset_path, "embeds.pkl")
    with open(name, 'wb') as f:
        pickle.dump({'text': all_embeds}, f)

    print(f"Embeddings saved to {name}")
    return name


def load_embeds(embed_path, device='cuda'):
    with open(embed_path, 'rb') as f:
        d = pickle.load(f)

    embeds_list = d['text']
    embeds = [x.to(device) if x.device != torch.device(device) else x for x in embeds_list]
    gen_embeds = torch.stack(embeds).squeeze()

    # d['text'] = [x.to('cuda') for x in d['text']]
    # gen_embeds = torch.stack(d['text']).squeeze()
    return gen_embeds


###########################embedding结束######################

###############2. 聚类 （k_dim个）###################################
def get_cluster_centers(embeds, k_dim):
    cluster_ids, cluster_centers = kmeans(
        embeds,
        num_clusters=k_dim,
        distance='cosine',
        device='cuda'
    )
    return cluster_ids, cluster_centers


# 重新排序聚类中心 注意这里没有重映射标签，因为我们并不需要
def sort_clusters(cluster_centers):
    num_centers = cluster_centers.size(0)
    device = cluster_centers.device

    # 计算 pairwise 距离（欧氏距离）
    dists = torch.cdist(cluster_centers, cluster_centers, p=2)  # shape: (k, k)
    visited = torch.zeros(num_centers, dtype=torch.bool, device=device)

    # 从任意一个起点开始（比如第 0 个）
    order = []
    current = 0
    visited[current] = True
    order.append(current)

    for _ in range(num_centers - 1):
        dists[current][visited] = float('inf')  # 避免回头
        next_idx = torch.argmin(dists[current]).item()
        visited[next_idx] = True
        order.append(next_idx)
        current = next_idx

    return torch.tensor(order, device=device)


#####################3. 分析########################
def analyze_cluster_similarity(embeddings, labels, cluster_centers):
    """分析类内和类间相似度"""
    # 确保所有张量在相同设备上
    device = embeddings.device
    labels = labels.to(device)
    cluster_centers = cluster_centers.to(device)

    # 类内相似度（每个样本与所属簇中心的相似度）
    intra_similarities = []
    for cluster_id in range(len(cluster_centers)):
        mask = labels == cluster_id
        cluster_embeds = embeddings[mask]
        if len(cluster_embeds) == 0:
            continue
        similarities = torch.nn.functional.cosine_similarity(
            cluster_embeds,
            cluster_centers[cluster_id].unsqueeze(0),
            dim=1
        )
        intra_similarities.append(similarities.mean().item())

    # 类间相似度优化（分批计算 + CPU计算）
    cluster_centers_cpu = cluster_centers.cpu().float()
    inter_total = 0.0
    inter_count = 0

    from tqdm import tqdm
    for i in tqdm(range(len(cluster_centers_cpu)), desc="计算类间相似度"):
        current = cluster_centers_cpu[i].unsqueeze(0)
        others = cluster_centers_cpu[i + 1:]
        similarities = torch.mm(current, others.T).squeeze(0)
        inter_total += similarities.sum().item()
        inter_count += len(others)

    inter_avg = inter_total / inter_count if inter_count > 0 else 0.0

    print(f"\n相似度分析结果:")
    print(f"总聚类数: {len(cluster_centers)} 个类")
    print(f"平均类内余弦相似度: {np.mean(intra_similarities):.4f}")
    print(f"最大类内相似度: {np.max(intra_similarities):.4f}")
    print(f"最小类内相似度: {np.min(intra_similarities):.4f}")
    print(f"平均类间相似度: {inter_avg:.4f}")


############################主函数####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--embedder_path', type=str, default="AbeHou/SemStamp-c4-sbert")
    parser.add_argument('--sp_dim', type=int, default=32)
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)

    # 检查现有嵌入文件
    embed_path = os.path.join(args.data_path, "embeds.pkl")
    if not os.path.exists(embed_path):
        # 当不存在时生成嵌入
        embed_path = embed_gen_list(args.data_path, args.embedder_path)
        print(f'Embedding generated at {embed_path}')
    else:
        print(f'Using existing embeddings at {embed_path}')

    # 聚类
    print("Generating cluster centers..")
    embeddings = load_embeds(embed_path)
    # if not embeddings.is_cuda:
    #     embeddings = embeddings.to('cuda')
    labels, cluster_centers = get_cluster_centers(embeddings, args.sp_dim)

    analyze_cluster_similarity(embeddings, labels, cluster_centers)

    # 原函数定义是sort_cluster
    sorted_indices = sort_clusters(cluster_centers)
    sorted_centers = cluster_centers[sorted_indices]
    torch.save(cluster_centers, f'{args.data_path}/cc.pt')