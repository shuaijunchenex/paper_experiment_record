import numpy as np

def generate_dirichlet_distribution(label_counts, num_clients, alpha):
    """
    label_counts: 每个类别的总数据量列表，例如 [2528, 6022]
    num_clients: 客户端数量，例如 10
    alpha: 狄利克雷分布参数
    """
    num_classes = len(label_counts)
    client_data = np.zeros((num_clients, num_classes), dtype=int)
    
    for k in range(num_classes):
        # 为当前类别生成 dirichlet 比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # 按照比例分配数据量
        counts = np.round(proportions * label_counts[k]).astype(int)
        
        # 修正四舍五入导致的误差，确保总数绝对一致
        diff = label_counts[k] - counts.sum()
        if diff > 0:
            # 随机把少的数据补给某些client
            counts[np.random.choice(num_clients, diff, replace=True)] += 1
        elif diff < 0:
            # 随机从某些有数据的client扣除多出的数据
            for _ in range(-diff):
                idx = np.random.choice(np.where(counts > 0)[0])
                counts[idx] -= 1
                
        client_data[:, k] = counts
        
    return client_data.tolist()

# 你的 CoLA 数据
labels_total = [2528, 6022]
clients = 10
alpha_value = 0.8  # 调整这个值：0.1 极度不均匀，1.0 适中，10 接近平均

result = generate_dirichlet_distribution(labels_total, clients, alpha_value)

print("custom: [")
for i, row in enumerate(result):
    print(f"   {row}{',' if i < len(result)-1 else ''}")
print("]")