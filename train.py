import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # [新增] 导入 tqdm

# 导入模块
from src.config import CONFIG
from src.dataset import RealDiseaseDataset
from src.model import HerbTransformerGenerator
from src.utils import generate_formula_sequence

def main():
    # 确保保存目录存在
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Step 1: Loading Herb Matrix...")
    try:
        try:
            df_herb = pd.read_csv(CONFIG['herb_file_path'], index_col=0, encoding='utf-8')
        except:
             df_herb = pd.read_csv(CONFIG['herb_file_path'], index_col=0, encoding='gbk')
        df_herb.fillna(0, inplace=True)
        target_pathways = df_herb.index.tolist()
        herb_names_list = df_herb.columns.tolist()
        herb_values = df_herb.T.values.astype(np.float32)
        
        if herb_values.max() != herb_values.min():
            herb_values = 2 * (herb_values - herb_values.min()) / (herb_values.max() - herb_values.min()) - 1
        
        herb_nes_tensor = torch.tensor(herb_values).to(CONFIG['device'])
        num_herbs, num_pathways = herb_nes_tensor.shape
        print(f"Matrix Loaded. Herbs: {num_herbs}, Pathways: {num_pathways}")
    except Exception as e:
        print(f"Error loading herb matrix: {e}")
        return

    print(f"Step 2: Loading Disease Data (Sparse)...")
    train_dataset = RealDiseaseDataset(CONFIG['disease_folder'], target_pathways, augment=True)
    val_dataset = RealDiseaseDataset(CONFIG['disease_folder'], target_pathways, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=4,      
        pin_memory=True,    
        persistent_workers=True
    )
    
    model = HerbTransformerGenerator(num_pathways, num_herbs, CONFIG).to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    print("Step 4: Start Training...")
    loss_history = []
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        total_treat_loss = 0
        
        # [修改] 使用 tqdm 包装 train_loader 以显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", unit="batch")
        
        for disease_batch in pbar:
            disease_batch = disease_batch.to(CONFIG['device'], non_blocking=True)
            optimizer.zero_grad()
            
            generated_effect, _, _ = generate_formula_sequence(
                model, disease_batch, herb_nes_tensor, CONFIG, hard=False
            )
            
            sick_mask = (torch.abs(disease_batch) > 0.01).float() 
            healthy_mask = 1.0 - sick_mask
            
            residual = disease_batch + generated_effect
            
            loss_treatment = torch.sum((residual * sick_mask) ** 2) / (sick_mask.sum() + 1e-6) * 100.0
            loss_side_effect = torch.sum((residual * healthy_mask) ** 2) / (healthy_mask.sum() + 1e-6) * 0.5
            loss_reg = torch.mean(generated_effect ** 2) * 0.001
            
            loss = loss_treatment + loss_side_effect + loss_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_treat_loss += loss_treatment.item()
            
            # [新增] 实时更新进度条上的 Loss 显示
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Treat': f"{loss_treatment.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 为了不让进度条和 Log 混在一起，这里使用 tqdm.write 打印
        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch} Summary | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                sample_disease = torch.tensor(val_dataset.data_matrix[0]).unsqueeze(0).to(CONFIG['device'])
                mask = (torch.abs(sample_disease) > 0.01).float()
                baseline_score = torch.sum((sample_disease * mask) ** 2).item()
                eff, indices, dosages = generate_formula_sequence(model, sample_disease, herb_nes_tensor, CONFIG, hard=True)
                residual_score = torch.sum(((sample_disease + eff) * mask)**2).item()
                
                indices = torch.stack(indices, dim=1).squeeze().cpu().numpy()
                dosages = torch.stack(dosages, dim=1).squeeze().cpu().numpy()
                predicted_herbs = [herb_names_list[i] for i in indices]
                
                tqdm.write(f"\n--- Epoch {epoch} Validation ---")
                tqdm.write(f"Sick Score (Before): {baseline_score:.4f}")
                tqdm.write(f"Sick Score (After):  {residual_score:.4f}")
                tqdm.write("Formula (Top 5):")
                if np.ndim(dosages) == 0:
                     tqdm.write(f"  {predicted_herbs[0]}: {dosages:.4f}")
                else:
                    for h, d in zip(predicted_herbs[:5], dosages[:5]):
                        tqdm.write(f"  {h}: {d:.4f}")
                tqdm.write("-" * 30 + "\n")

    save_path = "checkpoints/transformer_herb_generator.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Model saved to {save_path}")
    
    try:
        plt.plot(loss_history)
        plt.savefig("loss_curve.png")
    except:
        pass

if __name__ == "__main__":
    main()