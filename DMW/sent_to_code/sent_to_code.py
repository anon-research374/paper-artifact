from typing import Optional
import torch
from sentence_transformers import SentenceTransformer
import os

torch.cuda.empty_cache()
# 全局变量避免重复加载
_EMBEDDER = None
_CLUSTER_CENTERS = None
_BIT_LENGTH = None


def initialize_resources(cc_path: str, embedder_path: str, bit_length: int = 3):
    """初始化模型和聚类中心"""
    global _EMBEDDER, _CLUSTER_CENTERS, _BIT_LENGTH
    _CLUSTER_CENTERS = torch.load(cc_path)
    _EMBEDDER = SentenceTransformer(embedder_path)
    _BIT_LENGTH = bit_length


#添加pairwise_cosine方法，替换sent_to_code方法
def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    # print(f"data1: {data1}")
    # print(f"data2: {data2}")
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

def sent_to_code(sent: str, device: str = 'cuda:3', margin: float = 0.001) -> Optional[str]:
    """
    句子嵌入后生成哈希码
    """
    if _EMBEDDER is None or _CLUSTER_CENTERS is None:
        raise ValueError("Resources not initialized. Call initialize_resources first.")

    # 生成嵌入
    embed = _EMBEDDER.encode(sent, convert_to_tensor=True).to(device)

    # 使用 cosine 距离计算
    distances = pairwise_cosine(embed.unsqueeze(0), _CLUSTER_CENTERS.to(device)).squeeze(0)

    # 找出前两个最近的中心
    ranked = torch.argsort(distances)
    closest, second_closest = ranked[0], ranked[1]

    # 判断 margin 条件
    if (distances[second_closest] - distances[closest]) > margin:
        num_clusters = _CLUSTER_CENTERS.shape[0]
        assert num_clusters % (2 ** _BIT_LENGTH) == 0, "Cluster数必须能被2^bit_length整除"
        l = num_clusters // (2 ** _BIT_LENGTH)
        info_id = closest.item() // l
        return format(info_id, f'0{_BIT_LENGTH}b')

    return None

def compute_cost_cosine(sent: str, device: str = 'cuda:0') -> float:
    if _EMBEDDER is None or _CLUSTER_CENTERS is None:
        raise ValueError("Resources not initialized. Call initialize_resources first.")

    # 生成嵌入
    embed = _EMBEDDER.encode(sent, convert_to_tensor=True).to(device)

    # 使用 cosine 距离计算
    distances = pairwise_cosine(embed.unsqueeze(0), _CLUSTER_CENTERS.to(device)).squeeze(0)
    return distances.min().item()


def compute_cost_norm(sent: str, device: str = 'cuda:0') -> float:
    """直接通过句子计算cost"""
    if _EMBEDDER is None or _CLUSTER_CENTERS is None:
        raise ValueError("Resources not initialized. Call initialize_resources first.")

    embed = _EMBEDDER.encode(sent, convert_to_tensor=True).to(device)
    distances = torch.norm(_CLUSTER_CENTERS.to(device) - embed, dim=1)
    return distances.min().item()


if __name__ == "__main__":
    # 命令行测试
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cc_path', type=str, default="./data/4_kmeans/cc.pt")
    parser.add_argument('--embedder_path', type=str, default="AbeHou/SemStamp-c4-sbert")
    parser.add_argument('--bit_length', type=int, default=3)
    args = parser.parse_args()

    initialize_resources(args.cc_path, args.embedder_path, args.bit_length)
    test_sentence = "This is a good test sentence."
    print(f"Cost: {compute_cost(test_sentence)}")
    print(f"Bitstring: {sent_to_code(test_sentence)}")