from torchvision.transforms import Resize, ToTensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability

transforms = [Resize((144, 144)), 
              ToTensor()] #les transfo effectués sur chaque image

class Compose(object): #transform image to the same sizes and transform to tensor
    def __init__(self, transforms) -> None:
        self.tansforms = transforms  #ou est la focntion transforms ?

    def __call__(self, *image, target) : #permet d'appeler objet comme des fonctions 
                                                        #exemple : si c = Compose()  -> on peut faire c() pour appeler cette focntion
        for t in self.transforms:  #pour chaque transfo dans transforms (ici on en a 2 - resize + ToTensor())
            print(t)
            image = t(image)
        return image, target
    


class LeafDataset(Dataset):           #transform data #add labels 
    def __init__(self) -> None:
        super().__init__()

        self.path = "dataset/plant-pathology-2020-fgvc7"

        self.path_dir_X = os.path.join(self.path, 'images')

        self.path_Y = os.path.join(self.path, 'train.csv')
        self.dataframe_Y = pd.read_csv(self.path_Y)
        self.labels = self.dataframe_Y.loc[:, 'healthy':'scab']

        self.transform = A.Compose([
        #A.RandomResizedCrop(height=height_image, width=width_image, p=1.0), #au lieu de 500 - sinon le cpu ne suit pas - et kill tout les kernels python
        A.Resize(height=height_image, width=width_image),
        A.Rotate(20, p=1.0), 
        A.Flip(p=1.0),
        A.Transpose(p=1.0), 
        A.Normalize(p=1.0),  
        ToTensorV2(p=1.0),
        ], p=1.0)

        self.len = len(self.dataframe_Y)

    def __getitem__(self, index): #on defini le dataset et les transfo ici - car quand je vais appeller le dataloader - ca va parcourir toutes le simages donc passé par getitem
        img_name = self.dataframe_Y.loc[index, 'image_id']   #image_id,healthy,multiple_diseases,rust,scab
                                                             #Train_0,0,0,0,1      -> dans train.csv on a le nom du fichier ex : df[0]['image_id] = Train_0
        img_path = f"{self.path_dir_X}/{img_name}.jpg"
        image = plt.imread(img_path)

        image = self.transform(image = image)['image'] #resize / normalized / ....  #on prend ["image"] car renvoi un dictionnaire a la base 
        
        #test pour voir image de sorti
        #permute_transfo_image = image.permute(1, 2, 0)   #pour pouvoir l'afficher en plotlib
        #plt.imshow(permute_transfo_image)
        #plt.show()

        label = torch.tensor(np.argmax(self.labels.loc[index,:].values))  #on obtient la label avec argmax
        #print(f'label : {label}')   #maintenant on aplus que la label et plus le tableau 
                                    #on peut maitnent calculer une loss - on pouvait pas avant avec array : tensor([0, 0, 1, 0])

        return image, label
    
    def __len__(self):
        return self.len
    

def PositionalEncoding(sequence_len, output_dim, n=10000):
    P = torch.zeros((sequence_len, output_dim))
    for k in range(sequence_len):
        for i in range(0, output_dim, 2):
            denominator = torch.tensor(n, dtype=torch.float).pow(2 * i / output_dim)
            P[k, i] = torch.sin(k / denominator)
            P[k, i + 1] = torch.cos(k / denominator)
    return P


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, emb_size):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.n_patches = (img_size // patch_size) * (img_size // patch_size)


        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels = self.n_patches, kernel_size=patch_size, stride=patch_size), 
                                            # image -> réduite a partir de convolution 
                                            # out_channels = self.n_patches    - on passe de 144*144 pixels a self.n_patches ici 18 *18
                                            #kernel_size  patch size -> comme ca on va mapeer dans l'espace vectoriell l'image - par rapport a ces patch  - donc le kernel resume le patch en une valeur - c'est pour ca qu'on aplique kernel_size = patch_size
                                            #stride c'est le pas - donc si = kerbel size -> on aura pas de chevauchement
        
            nn.Flatten(start_dim=2), #1ere dimenson = taille du batch  - donc on commence a 2 
            nn.Linear(self.n_patches, emb_size) 
                                            #Flat -> transfo en sequence
                                            #n_patches -> emb_size     - on reduit le vecteur a la taille de emb size = tailled de chaque vector representatif d'image
        )
        
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
                                #classifier token

        self.pos_encoding = PositionalEncoding(sequence_len=self.n_patches, output_dim=emb_size)
                        #output_dim = dimension de l'encodage de position
                        #doit etre = dimension de sortie de la convolution du patch embeding

    def forward(self, x):
        #x = input image
        
        B, C, H, W = x.shape #batch size, channels, height, and width of the input image

        x = self.patch_embedding(x) #patch embeding layer
        #print(x.shape)

        x = x + self.pos_encoding.to(device)  # Apply positional encoding

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1) # Expand class token for each item in the batch
        x = torch.cat((cls_tokens, x), dim=1) # Shape: [B, n_patches+1, emb_size]

        

        return x
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output

#partie gauche (encoder) de l'architexture d'un transformer

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()

        self.attention = Attention(embed_dim,num_heads)
        self.norm1 = nn.LayerNorm(embed_dim) # first layer normalization - pour la premiere connexion résiduel

        self.mlp = nn.Sequential(  #MLP -> feed-forward layer
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim) #  second layer normalization - connexion résiduel aprés le MLP

        self.dropout = nn.Dropout(0.1) #dropout layer to prevent overfitting during training

    def forward(self, x):
        attn_out = self.attention(x)

        x = x + self.dropout(attn_out)  #dropout pour eviter overfiting    #ADD
        x = self.norm1(x)  #1ere normalisation                             #& NORM

        mlp_out = self.mlp(x)                                              #FEED-FORWARD

        x = x + self.dropout(mlp_out)                                      #ADD
        x = self.norm2(x) #2eme normalisation                              #& NORM

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.transformer_encoders = nn.Sequential( #-> création de la qéquance d'encoder        -> num_layers = num encoders
            *[TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #class token pour l'image en input 

        self.norm = nn.LayerNorm(embed_dim) #final layer normalization to stabilize the output of the Transformer Encoder layers

        # Define the classification head, which is a linear layer that maps the class token to the number of output classes
        self.head = nn.Linear(embed_dim, num_classes)   

    def forward(self, x):
        x = self.patch_embed(x)  #embed de l'image input 

        x = self.transformer_encoders(x) #image dasn la séquence de transformer

        cls_token = x[:, 0] # Extract the class token from the output sequence

        cls_token = self.norm(cls_token) # Apply layer normalization to the class token

        logits = self.head(cls_token) # Pass the class token through the classification head to compute the logits for each class

        return logits
    

    
def train(model, dataloader, criterion, optimizer, device):
    model.train()  #training mode 
    
    total_loss = 0
    correct = 0  

    for batch_idx, (data, target) in enumerate(dataloader):  

        data, target = data.to(device), target.to(device)  
        optimizer.zero_grad()  
        
        output = model(data) # Forward pass
        loss = criterion(output, target) 
        loss.backward() 
        total_loss += loss.item()  
        pred = output.argmax(dim=1, keepdim=True)  
        optimizer.step() 
        scheduler.step()

        correct += pred.eq(target.view_as(pred)).sum().item()  #number of correct prediction
        
        
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

if __name__ == '__main__':
    c = Compose(transforms)

    height_image, width_image = 144, 144

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    leaf_dataset = LeafDataset()
    batch_size=64

    train_loader = DataLoader(leaf_dataset, batch_size=batch_size, shuffle=True)

    model = VisionTransformer(
        img_size=144,
        patch_size=12,
        in_channels=3,
        embed_dim=128,
        num_heads=64,
        mlp_dim=128,
        num_layers=6,
        num_classes=4,
        device=device
    ).to(device)

    epochs = 10000

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = OneCycleLR(optimizer, max_lr=3e-4, total_steps=int(((len(leaf_dataset) - 1) // batch_size + 1) * epochs))
 
    train_losses = []


    for epoch in range(epochs): #training loop
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)

        train_losses.append(round(train_loss, 3))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Save the model at the end of each epoch
        if epoch%3 == 0:
            model_checkpoint = os.path.join("saved_models", f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), model_checkpoint)