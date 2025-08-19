# LLMs-LncRNA: Leveraging LLMs for LncRNA Classification and LncRNA-Protein Interaction Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-red.svg)](https://pytorch.org/)

A novel approach for sequence-based feature extraction using Large Language Models (LLMs) for Long non-coding RNA (LncRNA) classification and LncRNA-protein interaction prediction.

## 📋 Table of Contents
- [Overview](#overview)

- [Features](#features)

- [Installation](#installation)

- [Datasets](#datasets)

- [Usage](#usage)

- [Citation](#citation)

  

## 🔬 Overview

This repository implements a comprehensive framework for:
1. **Task 1**: LncRNA classification into functional categories (Antisense, Exonic, Linc, Sense No Exonic)
2. **Task 2**: LncRNA-protein interaction prediction

Our approach leverages pre-trained language models specifically designed for RNA sequences (RNA-FM) and protein sequences (ESM) to extract meaningful biological features for downstream prediction tasks.

## ✨ Features

- **Dual-task framework**: Supports both LncRNA classification and interaction prediction
- **Pre-trained model integration**: Utilizes RNA-FM for RNA sequences and ESM-2 for protein sequences
- **Multiple datasets**: Support for various benchmark datasets (RPI488, RPI369, RPI1807, RPI2241, NPInter)
- **Comprehensive evaluation**: Detailed performance metrics and visualization

## 🛠 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- conda or pip package manager

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/kelly159951/LLMs-LncRNA.git
cd LLMs-LncRNA
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yaml
conda activate RNA-PRO
```

Or install dependencies manually:
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install biopython pandas numpy tqdm matplotlib
pip install fair-esm
pip install fm-rna  # RNA-FM model
```

## 📊 Datasets

### Task 1: LncRNA Classification
- **Input**: LncRNA sequences in FASTA format (obtained through web crawling)
- **Labels**: Four functional categories (Antisense, Exonic, Linc, Sense No Exonic)
- **Files**: 
  - `lnc_class_new.fasta`: RNA sequences
  - `lnc_class_new_label.csv`: Corresponding labels

### Task 2: LncRNA-Protein Interaction Prediction
- **Positive pairs**: Known LncRNA-protein interactions (obtained through web crawling)
- **Negative pairs**: Generated non-interacting pairs (obtained through web crawling)
- **Files**:
  - `task2_rna_protein_pairs.txt`: Positive interaction pairs
  - `task2_nonrna_protein_pairs.txt`: Negative interaction pairs
  - `task2_rna_sequences.fasta`: RNA sequences
  - `task2_pro_sequences.fasta`: Protein sequences

### Benchmark Datasets
- RPI488, RPI369, RPI1807, RPI2241: Standard RNA-protein interaction datasets
- NPInter: Non-coding RNA interaction database (data acquired through automated crawling)

## 🚀 Usage

### Quick Start

#### Task 1: LncRNA Classification
```bash
# Basic training
CUDA_VISIBLE_DEVICES=0 python task.py \
  --task_num 1 \
  --rna_seq_file='./dataset/lnc_class_new.fasta' \
  --label_file='./dataset/lnc_class_new_label.csv' \
  --lr=0.00002 \
  --bs=32 \
  --epochs=50 \
  --use_pool
```

#### Task 2: LncRNA-Protein Interaction Prediction
```bash
# Basic training
CUDA_VISIBLE_DEVICES=0 python task.py \
  --task_num 2 \
  --pair_file='./dataset/task2_rna_protein_pairs.csv' \
  --rna_sequence_file='./dataset/task2_rna_sequences.fasta' \
  --pro_sequence_file='./dataset/task2_pro_sequences.fasta' \
  --non_pair_file='./dataset/task2_nonrna_protein_pairs.csv' \
  --non_rna_sequence_file='./dataset/task2_nonrna_sequences.fasta' \
  --non_pro_sequence_file='./dataset/task2_nonpro_sequences.fasta' \
  --lr=0.0001 \
  --bs=32 \
  --epochs=40 \
  --use_pool
```

### Advanced Usage

#### Multi-GPU Training
```bash
# Distributed training with multiple GPUs
torchrun --standalone --nproc_per_node=gpu task.py \
  --task_num 1 \
  --rna_seq_file='./dataset/lnc_class_new.fasta' \
  --label_file='./dataset/lnc_class_new_label.csv' \
  --lr=0.00002 \
  --bs=32 \
  --use_pool
```

#### Baseline Methods
```bash
# Run baseline methods with traditional feature extraction
python task.py \
  --baseline \
  --task_num 1 \
  --encode \
  --encoded_rna_file='./dataset/baseline_task1_encoded_data.csv'
```

#### Pipeline Training
```bash
# Use individual pipeline scripts
python task1_pipeline.py --lr=0.0001 --bs=32 --epochs=30
python task2_pipeline.py --lr=0.0001 --bs=32 --epochs=30
```

### Parameters

#### Common Parameters
- `--task_num`: Task number (1 for classification, 2 for interaction prediction)
- `--lr`: Learning rate (default: 0.0001)
- `--bs`: Batch size (default: 32)
- `--epochs`: Number of training epochs
- `--dropout`: Dropout rate (default: 0.1)
- `--use_pool`: Enable attention pooling
- `--disable_layer`: Number of layers to disable in pre-trained models
- `--device`: GPU device ID

#### Task-specific Parameters
**Task 1**:
- `--rna_seq_file`: Path to RNA sequences FASTA file
- `--label_file`: Path to labels CSV file
- `--inbalance_frac`: Class imbalance handling factor

**Task 2**:
- `--pair_file`: Path to positive pairs file
- `--non_pair_file`: Path to negative pairs file
- `--rna_sequence_file`: Path to RNA sequences
- `--pro_sequence_file`: Path to protein sequences



## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex

  title={Leveraging LLMs for LncRNA Classification and LncRNA-Protein Interaction Prediction: A Novel Approach for Sequence-Based Feature Extraction}
  #To bu published
```



## 🙏 Acknowledgments

- RNA-FM team for the pre-trained RNA foundation model
- ESM team for the protein language models
- BioPython community for sequence processing tools
- PyTorch team for the deep learning framework

## 📞 Contact

- **Author**: Yuhan Sun
- **Email**: kellysun@sjtu.edu.cn
- **Project Link**: [https://github.com/kelly159951/LLMs-LncRNA](https://github.com/kelly159951/LLMs-LncRNA)
