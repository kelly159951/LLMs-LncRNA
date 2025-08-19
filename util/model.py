import fm
import esm  
import torch
import torch.nn as nn
# model
class task1_Model(nn.Module):
    def __init__(self, dropout=0.0, attn_pool=False):
        super(task1_Model, self).__init__()
        self.rna_model,self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.mlp = nn.Sequential(
            nn.Linear(640, 640),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(640, 320),
            nn.GELU(),
            nn.Linear(320, 4)
        )
        if attn_pool:#Optional attention pooling layer
            self.pool_query = nn.Parameter(torch.randn(1, 1, 640))
            self.pooler = nn.MultiheadAttention(
                640, 8, dropout=dropout, batch_first=True
            )
        self.attn_pool = attn_pool

    def forward(self, batch_tokens):
        hidden = self.rna_model(batch_tokens, repr_layers=[12])["representations"][12]
        if self.attn_pool:
            batch_size = hidden.shape[0]
            padding_mask = batch_tokens == self.rna_model.padding_idx
            pk = self.pool_query.repeat(batch_size, 1, 1)
            feat, _ = self.pooler(
                query=pk, value=hidden, key=hidden,
                key_padding_mask=padding_mask
            )
            return self.mlp(feat.squeeze(dim=1))
        else:
            return self.mlp(hidden[:, 0])

class task2_Model(nn.Module):
    def __init__(self, dropout=0.0, attn_pool=False):
        super(task2_Model, self).__init__()
        self.rna_model, self.rna_alphabet = fm.pretrained.rna_fm_t12()
        self.pro_model, self.pro_alphabet = esm.pretrained.esm2_t33_650M_UR50D()  

        self.rna_batch_converter = self.rna_alphabet.get_batch_converter()
        self.pro_batch_converter = self.pro_alphabet.get_batch_converter()  
        self.mlp = nn.Sequential(
            nn.Linear(1920, 640),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(640, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        if attn_pool:#Optional attention pooling layer
            self.rna_pool_query = nn.Parameter(torch.randn(1, 1, 640))
            self.pro_pool_query = nn.Parameter(torch.randn(1, 1, 1280))
            self.rna_pooler = nn.MultiheadAttention(
                640, 8, dropout=dropout, batch_first=True
            )
            self.pro_pooler = nn.MultiheadAttention(
                1280, 8, dropout=dropout, batch_first=True
            )
        self.attn_pool = attn_pool

    def forward(self, batch_tokens):
        rna_tokens, pro_tokens = batch_tokens
        rna_hidden = self.rna_model(rna_tokens, repr_layers=[12])["representations"][12]
        pro_hidden = self.pro_model(pro_tokens, repr_layers=[33])["representations"][33]  

        if self.attn_pool:
            batch_size = rna_hidden.shape[0]
            rna_padding_mask = rna_tokens == self.rna_model.padding_idx
            rna_pk = self.rna_pool_query.repeat(batch_size, 1, 1)
            rna_feat, _ = self.rna_pooler(
                query=rna_pk, value=rna_hidden, key=rna_hidden,
                key_padding_mask=rna_padding_mask
            )
            rna_feat=rna_feat.squeeze(dim=1)
            pro_padding_mask = pro_tokens == self.pro_model.padding_idx
            pro_pk = self.pro_pool_query.repeat(batch_size, 1, 1)
            pro_feat, _ = self.pro_pooler(
                query=pro_pk, value=pro_hidden, key=pro_hidden,
                key_padding_mask=pro_padding_mask
            )
            pro_feat=pro_feat.squeeze(dim=1)
            feat=torch.cat([rna_feat, pro_feat], dim=-1)
            return self.mlp(feat)
        else:
            #Print shape of rna_hidden and pro_hidden
            # print(rna_hidden.shape)
            # print(pro_hidden.shape)
            hidden = torch.cat([rna_hidden[:,0], pro_hidden[:,0]], dim=-1)  # [batch, seq_len+2, 1920]
            return self.mlp(hidden) 
class baseline_task1_Model(nn.Module):
    def __init__(self,dropout):
        super(baseline_task1_Model, self).__init__()
        ae_ehiddim_1 = 45
        ae_ehiddim_2 = 32
        ae_ehiddim_3 = 16
        # Embedded feature dimension
        dim_embed = 100

        self.encoder = nn.Sequential(
            nn.Conv1d(1, ae_ehiddim_1, 3, padding=1),
            nn.BatchNorm1d(ae_ehiddim_1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(ae_ehiddim_1, ae_ehiddim_2, 3, padding=1),
            nn.BatchNorm1d(ae_ehiddim_2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(ae_ehiddim_2, ae_ehiddim_3, 3, padding=1),
            nn.BatchNorm1d(ae_ehiddim_3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(ae_ehiddim_3 * 42 * 1, dim_embed)

        # Decoder
        self.fc2 = nn.Linear(dim_embed, ae_ehiddim_3 * 42 * 1)
        self.unflatten = nn.Unflatten(1, (ae_ehiddim_3, 42, 1))

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ae_ehiddim_3, ae_ehiddim_3, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_ehiddim_3),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_ehiddim_3, ae_ehiddim_2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_ehiddim_2),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_ehiddim_2, ae_ehiddim_1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_ehiddim_1),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_ehiddim_1, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        num_classes = 4
        input_size = 100
        ln_1 = 512
        ln_2 = 64
        # Encoder
        self.fc = nn.Sequential(
            nn.Linear(input_size, ln_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ln_1, ln_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ln_2, num_classes),
            # nn.Softmax(dim=1)
        )

    def rna_hidden(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc1(x)
        encoded = x
        x = self.fc2(x)
        x = self.unflatten(x)
        x = x.squeeze(-1)  # Remove the last dimension to make input 3D
        x = self.decoder(x)
        return x, encoded
    
    def forward(self, x):
        x, encoded = self.rna_hidden(x)
        clas = self.fc(encoded)
        return clas, x

class baseline_task2_Model(nn.Module):
    def __init__(self,dropout):
        super(baseline_task2_Model, self).__init__()
        ae_rna_ehiddim_1 = 45
        ae_rna_ehiddim_2 = 32
        ae_rna_ehiddim_3 = 16
        # Embedded feature dimension
        dim_rna_embed = 100
        #rna autoencoder
        self.rna_encoder = nn.Sequential(
            nn.Conv1d(1, ae_rna_ehiddim_1, 3, padding=1),
            nn.BatchNorm1d(ae_rna_ehiddim_1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(ae_rna_ehiddim_1, ae_rna_ehiddim_2, 3, padding=1),
            nn.BatchNorm1d(ae_rna_ehiddim_2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(ae_rna_ehiddim_2, ae_rna_ehiddim_3, 3, padding=1),
            nn.BatchNorm1d(ae_rna_ehiddim_3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        self.rna_flatten = nn.Flatten()
        self.rna_fc1 = nn.Linear(ae_rna_ehiddim_3 * 42 * 1, dim_rna_embed)

        # Decoder
        self.rna_fc2 = nn.Linear(dim_rna_embed, ae_rna_ehiddim_3 * 42 * 1)
        self.rna_unflatten = nn.Unflatten(1, (ae_rna_ehiddim_3, 42, 1))

        self.rna_decoder = nn.Sequential(
            nn.ConvTranspose1d(ae_rna_ehiddim_3, ae_rna_ehiddim_3, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_rna_ehiddim_3),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_rna_ehiddim_3, ae_rna_ehiddim_2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_rna_ehiddim_2),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_rna_ehiddim_2, ae_rna_ehiddim_1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_rna_ehiddim_1),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_rna_ehiddim_1, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )


        #pro autoencoder
        # Autoencoder dimensions
        ae_pro_ehiddim_1 = 45
        ae_pro_ehiddim_2 = 32
        ae_pro_ehiddim_3 = 16
        dim_pro_embed = 100

        # Encoder
        self.pro_encoder = nn.Sequential(
            nn.Conv1d(1, ae_pro_ehiddim_1, 3, padding=3),
            nn.BatchNorm1d(ae_pro_ehiddim_1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(ae_pro_ehiddim_1, ae_pro_ehiddim_2, 3, padding=1),
            nn.BatchNorm1d(ae_pro_ehiddim_2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(ae_pro_ehiddim_2, ae_pro_ehiddim_3, 3, padding=1),
            nn.BatchNorm1d(ae_pro_ehiddim_3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)   # Reduces length by 2   
        )

        self.pro_flatten = nn.Flatten()
        self.pro_fc1 = nn.Linear(ae_pro_ehiddim_3 * 24, dim_pro_embed)  # Input dimension corrected to 320
        self.pro_fc2 = nn.Linear(dim_pro_embed, ae_pro_ehiddim_3 * 24)  # Output dimension corrected to 320
        self.pro_unflatten = nn.Unflatten(1, (ae_pro_ehiddim_3, 24, 1))

        self.pro_decoder = nn.Sequential(
            nn.ConvTranspose1d(ae_pro_ehiddim_3, ae_pro_ehiddim_3, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_pro_ehiddim_3),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_pro_ehiddim_3, ae_pro_ehiddim_2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_pro_ehiddim_2),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_pro_ehiddim_2, ae_pro_ehiddim_1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ae_pro_ehiddim_1),
            nn.ReLU(),

            nn.ConvTranspose1d(ae_pro_ehiddim_1, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )



        num_classes = 4
        input_size = dim_rna_embed + dim_pro_embed
        ln_1 = 512
        ln_2 = 64
        # mlp
        self.fc = nn.Sequential(
            nn.Linear(input_size, ln_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ln_1, ln_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ln_2, num_classes),
            # nn.Softmax(dim=1)
        )

    def rna_hidden(self, x):
        x = self.rna_encoder(x)
        x = self.rna_flatten(x)
        x = self.rna_fc1(x)
        encoded = x
        x = self.rna_fc2(x)
        x = self.rna_unflatten(x)
        x = x.squeeze(-1)  # Remove the last dimension to make input 3D
        x = self.rna_decoder(x)
        return x, encoded
    def pro_hidden(self, x):
        # print(x.shape)
        x = self.pro_encoder(x)
        # print(x.shape)
        x = self.pro_flatten(x)
        # print(x.shape)
        x = self.pro_fc1(x)
        # print(x.shape)
        encoded = x
        x = self.pro_fc2(x)
        # print(x.shape)
        x = self.pro_unflatten(x)
        # print(x.shape)
        x = x.squeeze(-1)  # Remove the last dimension to make input 3D
        # print(x.shape)
        x = self.pro_decoder(x)
        # print(x.shape)
        x=x[:,:,:188]
        # print(x.shape)
        return x, encoded
    
    def forward(self, rna, pro):
        rna, encoded_rna = self.rna_hidden(rna)
        pro, encoded_pro = self.pro_hidden(pro)
        x = torch.cat([encoded_rna, encoded_pro], dim=-1)
        x = self.fc(x)
        return x, rna, pro

    
def create_model(args):
    if args.baseline:
        if args.task_num==1:
            model = baseline_task1_Model(args.dropout).to(args.device)
        else:
            model = baseline_task2_Model(args.dropout).to(args.device)
    else:
        if(args.task_num==1):
            model = task1_Model(args.dropout, args.use_pool).to(args.device)
        else:
            model = task2_Model(args.dropout, args.use_pool).to(args.device)

    if args.checkpoint != '':
        weight = torch.load(args.checkpoint, map_location=args.device)
        # Remove 'module.' prefix from each layer in model (if any)
        new_state_dict = {}
        for k, v in weight.items():
            # If key name starts with 'module.', remove prefix
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print(f"Load model from {args.checkpoint}")
    return model
        