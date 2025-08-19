import os
import random

def read_fasta(filename):
    """读取FASTA文件，返回ID到序列的字典，并过滤长度超过1022的序列"""
    seqs = {}
    current_id = None
    current_seq = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None and len(''.join(current_seq)) <= 1022:
                    seqs[current_id] = ''.join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None and len(''.join(current_seq)) <= 1022:
            seqs[current_id] = ''.join(current_seq)
    return seqs

def ensure_dir(file_path):
    """确保文件路径中的目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def process_pairs(pairs, pair_filename, pro_filename, rna_filename, protein_seqs, rna_seqs, label):
    """处理并保存有效的配对数据"""
    valid_pairs = []
    
    # 过滤无效的配对
    for pro_id, rna_id in pairs:
        if pro_id in protein_seqs and rna_id in rna_seqs:
            valid_pairs.append((pro_id, rna_id))
        else:
            print(f"警告：忽略无效配对 ({pro_id}, {rna_id})，缺少对应序列")

    # 确保输出目录存在
    for path in [pair_filename, pro_filename, rna_filename]:
        ensure_dir(path)

    # 保存CSV配对文件（新增标题行）
    with open(pair_filename, 'w') as f:
        f.write("proSeqname,Seqname,Label\n")  # CSV标题
        for pro_id, rna_id in valid_pairs:
            f.write(f"{pro_id},{rna_id},{label}\n")

    # 保存蛋白质序列（保持原格式）
    with open(pro_filename, 'w') as f:
        for pro_id, _ in valid_pairs:
            f.write(f">{pro_id}\n{protein_seqs[pro_id]}\n")

    # 保存RNA序列（保持原格式）
    with open(rna_filename, 'w') as f:
        for _, rna_id in valid_pairs:
            f.write(f">{rna_id}\n{rna_seqs[rna_id]}\n")

    return valid_pairs

def balance_pairs(pos_pairs, neg_pairs, protein_seqs, rna_seqs):
    """平衡正负反应对的数量"""
    pos_count = len(pos_pairs)
    neg_count = len(neg_pairs)

    if pos_count > neg_count:
        # 从所有可能的序列中随机抽取新的负反应对
        all_pro_ids = list(protein_seqs.keys())
        all_rna_ids = list(rna_seqs.keys())
        existing_pairs = set(pos_pairs + neg_pairs)

        while pos_count > neg_count:
            pro_id = random.choice(all_pro_ids)
            rna_id = random.choice(all_rna_ids)
            if (pro_id, rna_id) not in existing_pairs and len(protein_seqs[pro_id]) <= 1022 and len(rna_seqs[rna_id]) <= 1022:
                neg_pairs.append((pro_id, rna_id))
                existing_pairs.add((pro_id, rna_id))
                neg_count += 1
    elif neg_count > pos_count:
        # 随机删除部分负反应对
        neg_pairs = random.sample(neg_pairs, pos_count)

    return pos_pairs, neg_pairs

def main():
    # 路径配置
    base_dir = "dataset/data"
    seq_dir = os.path.join(base_dir, "sequence")
    datas=['RPI488','RPI369','RPI1807','RPI2241','NPInter']
    for data in datas:
        print(f"处理数据集：{data}")
        # 读取配对数据（改用制表符分割）
        pairs = []
        with open(os.path.join(base_dir, data+"_pairs.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')  # 明确使用制表符分割
                if len(parts) >= 3:
                    pairs.append((parts[0], parts[1], parts[2]))

        # 分割正负样本
        pos_pairs = [(p[0], p[1]) for p in pairs if p[2] == '1']
        neg_pairs = [(p[0], p[1]) for p in pairs if p[2] == '0']

        # 读取序列数据
        protein_seqs = read_fasta(os.path.join(seq_dir, data+"_protein_seq.fa"))
        rna_seqs = read_fasta(os.path.join(seq_dir, data+"_rna_seq.fa"))

        # 处理正样本（改为CSV扩展名）
        pos_pairs = process_pairs(
            pos_pairs,
            os.path.join(base_dir, data+"_pos_pairs.csv"),  # 改为csv
            os.path.join(seq_dir, data+"_pos_pro_seq.fa"),
            os.path.join(seq_dir, data+"_pos_rna_seq.fa"),
            protein_seqs, rna_seqs, '1'
        )

        # 处理负样本（改为CSV扩展名）
        neg_pairs = process_pairs(
            neg_pairs,
            os.path.join(base_dir, data+"_neg_pairs.csv"),  # 改为csv
            os.path.join(seq_dir, data+"_neg_pro_seq.fa"),
            os.path.join(seq_dir, data+"_neg_rna_seq.fa"),
            protein_seqs, rna_seqs, '0'
        )

        # 平衡正负反应对
        print(f"平衡正负反应对数量：{len(pos_pairs)} 正对 vs. {len(neg_pairs)} 负对")
        pos_pairs, neg_pairs = balance_pairs(pos_pairs, neg_pairs, protein_seqs, rna_seqs)
        print(f"平衡后：{len(pos_pairs)} 正对 vs. {len(neg_pairs)} 负对")
        # 重新保存平衡后的正负反应对
        process_pairs(
            pos_pairs,
            os.path.join(base_dir, data+"_pos_pairs_balanced.csv"),
            os.path.join(seq_dir, data+"_pos_pro_seq_balanced.fa"),
            os.path.join(seq_dir, data+"_pos_rna_seq_balanced.fa"),
            protein_seqs, rna_seqs, '1'
        )

        process_pairs(
            neg_pairs,
            os.path.join(base_dir, data+"_neg_pairs_balanced.csv"),
            os.path.join(seq_dir, data+"_neg_pro_seq_balanced.fa"),
            os.path.join(seq_dir, data+"_neg_rna_seq_balanced.fa"),
            protein_seqs, rna_seqs, '0'
        )

if __name__ == "__main__":
    main()