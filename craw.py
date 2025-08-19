import requests
from bs4 import BeautifulSoup
from Bio import SeqIO
from tqdm import tqdm  # 导入 tqdm，用于显示进度条
import os
import csv
from concurrent.futures import ThreadPoolExecutor


# 定义一个函数来发起请求并解析预测结果
def get_prediction(rna_seq, protein_seq):
    # 准备 POST 请求的表单数据
    data = {
        "r_input": str(rna_seq.seq),  # RNA 序列
        "p_input": str(protein_seq.seq)  # 蛋白质序列
    }
    
    # 发起 POST 请求
    try:
        response = requests.post(url, data=data)
        # 解析返回的 HTML 内容
        soup = BeautifulSoup(response.text, 'lxml')  # 使用 lxml 解析器
        
        # 初始化预测值
        rf_prediction = None
        svm_prediction = None
        
        # 查找预测结果
        for td in soup.find_all("td"):
            if "Prediction using RF classifier" in td.text:
                rf_prediction = td.find_next("td").find_next("td").text.strip()
            elif "Prediction using SVM classifier" in td.text:
                svm_prediction = td.find_next("td").find_next("td").text.strip()
        
        return rf_prediction, svm_prediction
    except Exception as e:
        print(f"Error with RNA {rna_seq.id} and Protein {protein_seq.id}: {e}")
        return None, None

# 定义一个保存结果的函数
def save_predictions_to_csv(predictions, output_file):
    # 确保目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RF Prediction', 'SVM Prediction'])

        # 写入预测结果
        for rf_pred, svm_pred in predictions:
            writer.writerow([rf_pred, svm_pred])
    
    print(f"Results saved to {output_file}")

# 使用多线程加速处理 RNA 和蛋白质序列的配对
def process_rna_protein_pairs():
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 创建任务并提交到线程池
        futures = []
        for rna, protein in zip(rna_sequences, protein_sequences):
            futures.append(executor.submit(get_prediction, rna, protein))
        
        # 获取每个任务的结果
        for future in tqdm(futures, desc="Processing RNA-Protein pairs", total=len(futures)):
            rf_prediction, svm_prediction = future.result()
            rf_predictions.append(rf_prediction)
            svm_predictions.append(svm_prediction)

# 处理 Non-RNA 和 Non-Protein 序列的配对
def process_non_rna_protein_pairs():
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 创建任务并提交到线程池
        futures = []
        for non_rna, non_protein in zip(non_rna_sequences, non_protein_sequences):
            futures.append(executor.submit(get_prediction, non_rna, non_protein))
        
        # 获取每个任务的结果
        for future in tqdm(futures, desc="Processing Non-RNA-Non-Protein pairs", total=len(futures)):
            rf_prediction, svm_prediction = future.result()
            rf_predictions.append(rf_prediction)
            svm_predictions.append(svm_prediction)
# 输入文件路径
rna_file = 'dataset/task2_rna_sequences.fasta'
protein_file = 'dataset/task2_pro_sequences.fasta'
non_rna_file = 'dataset/task2_nonrna_sequences.fasta'
non_protein_file = 'dataset/task2_nonpro_sequences.fasta'

# 读取 RNA 和蛋白质序列
rna_sequences = list(SeqIO.parse(rna_file, "fasta"))
protein_sequences = list(SeqIO.parse(protein_file, "fasta"))
non_rna_sequences = list(SeqIO.parse(non_rna_file, "fasta"))
non_protein_sequences = list(SeqIO.parse(non_protein_file, "fasta"))

# 提交数据到 RPISeq 结果页面
url = "http://pridb.gdcb.iastate.edu/RPISeq/results.php"

# 存储预测结果
rf_predictions = []  # 用于存储 RF 分类器预测概率
svm_predictions = []  # 用于存储 SVM 分类器预测概率

# 运行 RNA 和蛋白质配对处理
process_rna_protein_pairs()
rna_protein_output_file = "result/rna_pro_predictions_rf_svm.csv"
save_predictions_to_csv(zip(rf_predictions, svm_predictions), rna_protein_output_file)

# 清空预测结果，以便存储 Non-RNA 和 Non-Protein 配对的结果
rf_predictions = []
svm_predictions = []

# 运行 Non-RNA 和 Non-Protein 配对处理
process_non_rna_protein_pairs()
non_rna_protein_output_file = "result/non_rna_pro_predictions_rf_svm.csv"
save_predictions_to_csv(zip(rf_predictions, svm_predictions), non_rna_protein_output_file)
