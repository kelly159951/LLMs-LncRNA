import os
import random

def read_fasta(filename):
    """Read FASTA file, return dictionary from ID to sequence, and filter sequences longer than 1022"""
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
    """Ensure the directory in the file path exists"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def process_pairs(pairs, pair_filename, pro_filename, rna_filename, protein_seqs, rna_seqs, label):
    """Process and save valid paired data"""
    valid_pairs = []
    
    # Filter invalid pairs
    for pro_id, rna_id in pairs:
        if pro_id in protein_seqs and rna_id in rna_seqs:
            valid_pairs.append((pro_id, rna_id))
        else:
            print(f"Warning: Ignoring invalid pair ({pro_id}, {rna_id}), missing corresponding sequence")

    # Ensure output directory exists
    for path in [pair_filename, pro_filename, rna_filename]:
        ensure_dir(path)

    # Save CSV pair file (add header row)
    with open(pair_filename, 'w') as f:
        f.write("proSeqname,Seqname,Label\n")  # CSV header
        for pro_id, rna_id in valid_pairs:
            f.write(f"{pro_id},{rna_id},{label}\n")

    # Save protein sequences (maintain original format)
    with open(pro_filename, 'w') as f:
        for pro_id, _ in valid_pairs:
            f.write(f">{pro_id}\n{protein_seqs[pro_id]}\n")

    # Save RNA sequences (maintain original format)
    with open(rna_filename, 'w') as f:
        for _, rna_id in valid_pairs:
            f.write(f">{rna_id}\n{rna_seqs[rna_id]}\n")

    return valid_pairs

def balance_pairs(pos_pairs, neg_pairs, protein_seqs, rna_seqs):
    """Balance the number of positive and negative reaction pairs"""
    pos_count = len(pos_pairs)
    neg_count = len(neg_pairs)

    if pos_count > neg_count:
        # Randomly select new negative reaction pairs from all possible sequences
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
        # Randomly remove some negative reaction pairs
        neg_pairs = random.sample(neg_pairs, pos_count)

    return pos_pairs, neg_pairs

def main():
    # Path configuration
    base_dir = "dataset/data"
    seq_dir = os.path.join(base_dir, "sequence")
    datas=['RPI488','RPI369','RPI1807','RPI2241','NPInter']
    for data in datas:
        print(f"Processing dataset: {data}")
        # Read paired data (use tab separator)
        pairs = []
        with open(os.path.join(base_dir, data+"_pairs.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')  # Explicitly use tab separator
                if len(parts) >= 3:
                    pairs.append((parts[0], parts[1], parts[2]))

        # Split positive and negative samples
        pos_pairs = [(p[0], p[1]) for p in pairs if p[2] == '1']
        neg_pairs = [(p[0], p[1]) for p in pairs if p[2] == '0']

        # Read sequence data
        protein_seqs = read_fasta(os.path.join(seq_dir, data+"_protein_seq.fa"))
        rna_seqs = read_fasta(os.path.join(seq_dir, data+"_rna_seq.fa"))

        # Process positive samples (change to CSV extension)
        pos_pairs = process_pairs(
            pos_pairs,
            os.path.join(base_dir, data+"_pos_pairs.csv"),  # change to csv
            os.path.join(seq_dir, data+"_pos_pro_seq.fa"),
            os.path.join(seq_dir, data+"_pos_rna_seq.fa"),
            protein_seqs, rna_seqs, '1'
        )

        # Process negative samples (change to CSV extension)
        neg_pairs = process_pairs(
            neg_pairs,
            os.path.join(base_dir, data+"_neg_pairs.csv"),  # change to csv
            os.path.join(seq_dir, data+"_neg_pro_seq.fa"),
            os.path.join(seq_dir, data+"_neg_rna_seq.fa"),
            protein_seqs, rna_seqs, '0'
        )

        # Balance positive and negative reaction pairs
        print(f"Balancing positive and negative reaction pairs: {len(pos_pairs)} positive vs. {len(neg_pairs)} negative")
        pos_pairs, neg_pairs = balance_pairs(pos_pairs, neg_pairs, protein_seqs, rna_seqs)
        print(f"After balancing: {len(pos_pairs)} positive vs. {len(neg_pairs)} negative")
        # Re-save balanced positive and negative reaction pairs
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