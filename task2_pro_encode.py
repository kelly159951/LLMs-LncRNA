import torch  
import esm  
  
# 加载预训练的ESM模型  
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  
model = model.eval()  
print(model)
  
# 准备数据  
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  
batch_converter = alphabet.get_batch_converter()  
batch_labels, batch_strs, batch_tokens = batch_converter([(None, sequence)])  

# 计算嵌入  
with torch.no_grad():  
    results = model(batch_tokens, repr_layers=[33])  
token_representations = results["representations"][33]  
  
# token_representations 现在包含了序列中每个氨基酸的嵌入
print(token_representations)
print(len(sequence))  #65
print(token_representations.shape) #[1, 67, 1280]