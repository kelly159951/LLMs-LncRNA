import torch  
import esm  
  
# Load pre-trained ESM model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  
model = model.eval()  
print(model)
  
# Prepare data
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  
batch_converter = alphabet.get_batch_converter()  
batch_labels, batch_strs, batch_tokens = batch_converter([(None, sequence)])  

# Calculate embeddings
with torch.no_grad():  
    results = model(batch_tokens, repr_layers=[33])  
token_representations = results["representations"][33]  
  
# token_representations now contains embeddings for each amino acid in the sequence
print(token_representations)
print(len(sequence))  #65
print(token_representations.shape) #[1, 67, 1280]