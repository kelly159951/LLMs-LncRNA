


#task1
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=gpu task.py  --task_num 1 --rna_seq_file='./dataset/lnc_class_new.fasta' --label_file='./dataset/lnc_class_new_label.csv' --lr=0.00002 --bs=32 --base_log='./debug' --inbalance_frac=1.2 --dropout=0.2 --epochs=50 --disable_layer=12 --num_worker=4 --use_pool 

CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=gpu task.py  --task_num 1 --rna_seq_file='./dataset/lnc_class_new.fasta' --label_file='./dataset/lnc_class_new_label.csv' --lr=0.00002 --bs=32 --base_log='./log' --inbalance_frac=1 --dropout=0.15 --epochs=50 --disable_layer=12 --num_worker=4 --use_pool
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=gpu task.py  --task_num 1 --rna_seq_file='./dataset/lnc_class_new.fasta' --label_file='./dataset/lnc_class_new_label.csv' --lr=0.00002 --bs=32 --base_log='./log' --inbalance_frac=1.2 --dropout=0.2 --epochs=50 --disable_layer=11 --num_worker=4 --use_pool  

#task2
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=gpu task.py --task_num 2 --pair_file='./dataset/task2_rna_protein_pairs.csv' --rna_sequence_file='./dataset/task2_rna_sequences.fasta' --pro_sequence_file='./dataset/task2_pro_sequences.fasta' --non_pair_file='./dataset/task2_nonrna_protein_pairs.csv' --non_rna_sequence_file='./dataset/task2_nonrna_sequences.fasta' --non_pro_sequence_file='./dataset/task2_nonpro_sequences.fasta' --lr=0.0001 --bs=16 --dropout=0.2 --epochs=40 --disable_layer=11 --base_log='./log' --num_worker=4  --use_pool

#task2 with checkpoint
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nproc_per_node=gpu task.py --task_num 2 --pair_file='./dataset/task2_rna_protein_pairs.vsc' --rna_sequence_file='./dataset/task2_rna_sequences.fasta' --pro_sequence_file='./dataset/task2_pro_sequences.fasta' --non_pair_file='./dataset/task2_nonrna_protein_pairs.csv' --non_rna_sequence_file='./dataset/task2_nonrna_sequences.fasta' --non_pro_sequence_file='./dataset/task2_nonpro_sequences.fasta' --lr=0.0001 --bs=32 --dropout=0.1 --epochs=30 --disable_layer=12 --base_log='./log' --num_worker=4  --use_pool --skip=11 --checkpoint='log/task_2_1732803970.351648/best_model.pth' --log_dir='log/task_2_1732803970.351648'

#baseline task1
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu task.py  --baseline --task_num 1 --rna_seq_file='./dataset/lnc_class_new.fasta' --label_file='./dataset/lnc_class_new_label.csv' --alpha=0.4 --encode --encoded_rna_file='./dataset/baseline_task1_encoded_data.csv' --lr=0.00002 --bs=32 --base_log='./log' --inbalance_frac=1 --dropout=0.2 --epochs=50  --num_worker=4 
#baseline task1 not encode
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu task.py  --baseline --task_num 1 --rna_seq_file='./dataset/lnc_class_new.fasta' --label_file='./dataset/lnc_class_new_label.csv' --alpha=0.4 --encoded_rna_file='./dataset/baseline_task1_encoded_data.csv' --lr=0.00002 --bs=32 --base_log='./log' --inbalance_frac=1 --dropout=0.2 --epochs=50 --num_worker=4 

#baseline task2
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=gpu task.py  --baseline --task_num 2 --pair_file='./dataset/task2_rna_protein_pairs.csv' --rna_sequence_file='./dataset/task2_rna_sequences.fasta' --pro_sequence_file='./dataset/task2_pro_sequences.fasta' --non_pair_file='./dataset/task2_nonrna_protein_pairs.csv' --non_rna_sequence_file='./dataset/task2_nonrna_sequences.fasta' --non_pro_sequence_file='./dataset/task2_nonpro_sequences.fasta' --lr=0.0001 --bs=32 --dropout=0.1 --epochs=100  --base_log='./log' --num_worker=4 --alpha=0.4 --encode --encoded_rna_file='./dataset/baseline_task2_encoded_rna_data.csv' --encoded_pro_file='./dataset/baseline_task2_encoded_pro_data.csv' --encoded_non_rna_file='./dataset/baseline_task2_encoded_non_rna_data.csv' --encoded_non_pro_file='./dataset/baseline_task2_encoded_non_pro_data.csv' --num_worker=4



#task2 other dataset 
#RPI1488
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=gpu task.py --task_num 2 --pair_file='./dataset/data/RPI488_pos_pairs.csv' --rna_sequence_file='./dataset/data/sequence/RPI488_pos_rna_seq.fa' --pro_sequence_file='./dataset/data/sequence/RPI488_pos_pro_seq.fa' --non_pair_file='./dataset/data/RPI488_neg_pairs.csv' --non_rna_sequence_file='./dataset/data/sequence/RPI488_neg_rna_seq.fa' --non_pro_sequence_file='./dataset/data/sequence/RPI488_neg_pro_seq.fa'  --lr=0.001 --bs=32 --dropout=0.2 --epochs=40 --disable_layer=12 --base_log='./log' --num_worker=4  --use_pool --random

#RPI369
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=gpu task.py --task_num 2 --pair_file='./dataset/data/RPI369_pos_pairs_balanced.csv' --rna_sequence_file='./dataset/data/sequence/RPI369_pos_rna_seq_balanced.fa' --pro_sequence_file='./dataset/data/sequence/RPI369_pos_pro_seq_balanced.fa' --non_pair_file='./dataset/data/RPI369_neg_pairs_balanced.csv' --non_rna_sequence_file='./dataset/data/sequence/RPI369_neg_rna_seq_balanced.fa' --non_pro_sequence_file='./dataset/data/sequence/RPI369_neg_pro_seq_balanced.fa'  --lr=0.0001 --bs=32 --dropout=0.2 --epochs=40 --disable_layer=12 --base_log='./log' --num_worker=4  --use_pool --random

#RPI1807
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=gpu task.py --task_num 2 --pair_file='./dataset/data/RPI1807_pos_pairs_balanced.csv' --rna_sequence_file='./dataset/data/sequence/RPI1807_pos_rna_seq_balanced.fa' --pro_sequence_file='./dataset/data/sequence/RPI1807_pos_pro_seq_balanced.fa' --non_pair_file='./dataset/data/RPI1807_neg_pairs_balanced.csv' --non_rna_sequence_file='./dataset/data/sequence/RPI1807_neg_rna_seq_balanced.fa' --non_pro_sequence_file='./dataset/data/sequence/RPI1807_neg_pro_seq_balanced.fa'  --lr=0.0001 --bs=32 --dropout=0.2 --epochs=40 --disable_layer=12 --base_log='./log' --num_worker=4  --use_pool --random

#RPI2241
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=gpu task.py --task_num 2 --pair_file='./dataset/data/RPI2241_pos_pairs_balanced.csv' --rna_sequence_file='./dataset/data/sequence/RPI2241_pos_rna_seq_balanced.fa' --pro_sequence_file='./dataset/data/sequence/RPI2241_pos_pro_seq_balanced.fa' --non_pair_file='./dataset/data/RPI2241_neg_pairs_balanced.csv' --non_rna_sequence_file='./dataset/data/sequence/RPI2241_neg_rna_seq_balanced.fa' --non_pro_sequence_file='./dataset/data/sequence/RPI2241_neg_pro_seq_balanced.fa'  --lr=0.0001 --bs=32 --dropout=0.2 --epochs=40 --disable_layer=12 --base_log='./log' --num_worker=4  --use_pool --random

#NPInter
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=gpu task.py --task_num 2 --pair_file='./dataset/data/NPInter_pos_pairs_balanced.csv' --rna_sequence_file='./dataset/data/sequence/NPInter_pos_rna_seq_balanced.fa' --pro_sequence_file='./dataset/data/sequence/NPInter_pos_pro_seq_balanced.fa' --non_pair_file='./dataset/data/NPInter_neg_pairs_balanced.csv' --non_rna_sequence_file='./dataset/data/sequence/NPInter_neg_rna_seq_balanced.fa' --non_pro_sequence_file='./dataset/data/sequence/NPInter_neg_pro_seq_balanced.fa'  --lr=0.0001 --bs=32 --dropout=0.2 --epochs=40 --disable_layer=12 --base_log='./log' --num_worker=4  --use_pool --random






#distilation 
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=gpu task.py  --task_num 1 --rna_seq_file='./dataset/sample.fasta' --label_file='./dataset/sample_label.csv' --lr=0.00002 --bs=32 --base_log='./distilation' --inbalance_frac=1.2 --dropout=0.2 --epochs=50 --disable_layer=12 --num_worker=4 --use_pool --distilation_rna_fm
