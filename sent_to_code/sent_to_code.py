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


# This function will be defined and used in the main block
def setup_nltk(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)
    success = nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    if not success:
        print(
            f"Please execute manually:\nmkdir -p {nltk_data_path} && python -m nltk.downloader -d {nltk_data_path} punkt")
        exit(1)
    nltk.data.path.insert(0, nltk_data_path)


warnings.filterwarnings("ignore")


def worker(rank, text_chunk, embedder_path, queue, encode_batch_size):
    device = f'cuda:{rank}'
    embedder = SentenceTransformer(embedder_path, device=device)
    embedder = embedder.eval()
    sent_embeds = []
    with tqdm(total=len(text_chunk), desc=f"Worker {rank} Encoding", position=rank) as pbar:
        for i in range(0, len(text_chunk), encode_batch_size):
            batch_texts = text_chunk[i:i + encode_batch_size]
            batch_embeds = embedder.encode(batch_texts, convert_to_tensor=True)
            sent_embeds.extend([e.cpu().numpy() for e in batch_embeds])
            pbar.update(len(batch_texts))
    queue.put(sent_embeds)


def embed_gen_list(dataset_path, embedder_path, encode_batch_size=32, num_gpus=torch.cuda.device_count()):
    from multiprocessing import Process, Queue

    dataset = load_from_disk(dataset_path)
    texts = dataset['text']
    flattened_texts = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        flattened_texts.extend(sentences)

    total_progress = tqdm(total=len(flattened_texts), desc="Total Progress", position=num_gpus)
    text_chunks = [flattened_texts[i::num_gpus] for i in range(num_gpus)]
    queue = Queue()
    processes = []

    for rank, text_chunk in enumerate(text_chunks):
        p = Process(target=worker, args=(rank, text_chunk, embedder_path, queue, encode_batch_size))
        p.start()
        processes.append(p)

    all_embeds = []
    while any(p.is_alive() for p in processes) or not queue.empty():
        while not queue.empty():
            chunk_embeds = queue.get()
            all_embeds.extend([torch.tensor(e).to('cuda') for e in chunk_embeds])
            total_progress.update(len(chunk_embeds))

    total_progress.close()
    for p in processes:
        p.join()

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
    return gen_embeds


def get_cluster_centers(embeds, k_dim):
    cluster_ids, cluster_centers = kmeans(
        embeds,
        num_clusters=k_dim,
        distance='cosine',
        device='cuda'
    )
    return cluster_ids, cluster_centers


def sort_clusters(cluster_centers):
    num_centers = cluster_centers.size(0)
    device = cluster_centers.device
    dists = torch.cdist(cluster_centers, cluster_centers, p=2)
    visited = torch.zeros(num_centers, dtype=torch.bool, device=device)
    order = []
    current = 0
    visited[current] = True
    order.append(current)
    for _ in range(num_centers - 1):
        dists[current][visited] = float('inf')
        next_idx = torch.argmin(dists[current]).item()
        visited[next_idx] = True
        order.append(next_idx)
        current = next_idx
    return torch.tensor(order, device=device)


def analyze_cluster_similarity(embeddings, labels, cluster_centers):
    device = embeddings.device
    labels = labels.to(device)
    cluster_centers = cluster_centers.to(device)
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

    cluster_centers_cpu = cluster_centers.cpu().float()
    inter_total = 0.0
    inter_count = 0
    for i in tqdm(range(len(cluster_centers_cpu)), desc="Calculating inter-cluster similarity"):
        current = cluster_centers_cpu[i].unsqueeze(0)
        others = cluster_centers_cpu[i + 1:]
        similarities = torch.mm(current, others.T).squeeze(0)
        inter_total += similarities.sum().item()
        inter_count += len(others)
    inter_avg = inter_total / inter_count if inter_count > 0 else 0.0

    print(f"\nSimilarity Analysis Results:")
    print(f"Total Clusters: {len(cluster_centers)}")
    print(f"Average Intra-cluster Cosine Similarity: {np.mean(intra_similarities):.4f}")
    print(f"Max Intra-cluster Similarity: {np.max(intra_similarities):.4f}")
    print(f"Min Intra-cluster Similarity: {np.min(intra_similarities):.4f}")
    print(f"Average Inter-cluster Similarity: {inter_avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--embedder_path', type=str, default="AbeHou/SemStamp-c4-sbert")
    parser.add_argument('--sp_dim', type=int, default=32)
    parser.add_argument('--nltk-data-path', type=str, default='./nltk_data', help='Path to download/load NLTK data.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_nltk(args.nltk_data_path)
    mp.set_start_method('spawn', force=True)

    embed_path = os.path.join(args.data_path, "embeds.pkl")
    if not os.path.exists(embed_path):
        embed_path = embed_gen_list(args.data_path, args.embedder_path)
        print(f'Embedding generated at {embed_path}')
    else:
        print(f'Using existing embeddings at {embed_path}')

    print("Generating cluster centers..")
    embeddings = load_embeds(embed_path, device=device)
    labels, cluster_centers = get_cluster_centers(embeddings, args.sp_dim)

    analyze_cluster_similarity(embeddings, labels, cluster_centers)

    sorted_indices = sort_clusters(cluster_centers)
    sorted_centers = cluster_centers[sorted_indices]
    torch.save(sorted_centers, f'{args.data_path}/cc.pt')