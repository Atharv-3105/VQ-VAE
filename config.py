import torch
class Config:
    def __init__(self):
        self.in_channels = 3
        self.hidden_channels = 128
        self.latent_dim = 256
        self.out_channels = 3
        self.img_size = 64 
        self.num_embeddings = 1024 #Total Number of Embeddings in the CodeBook
        self.embedding_dim = 256   #Dimensionality of Each CodeBook Vector
        self.beta = 0.25           #Commitment Loss Weight
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.epochs = 20
        self.lr = 2e-4
        self.project_name = "VQ_VAE"
        self.save_dir = "./saves"
        self.log_interval = 100
        self.save_interval = 1
        self.use_wandb = True
        