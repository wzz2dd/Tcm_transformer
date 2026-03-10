import torch

CONFIG = {
    'd_model': 256,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.1,
    'max_seq_len': 12,       
    'batch_size': 64,
    
    'epochs': 1000,
    'lr': 1e-3,  
    
    'aug_noise': 0.05,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # [路径修改] 指向根目录下的 data 文件夹
    'herb_file_path': "data/通路-中药组合_NES矩阵.csv", 
    'disease_folder': "data/disease_data_folder"      
}