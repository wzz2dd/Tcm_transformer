import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats  # 确保安装了 scipy: pip install scipy

# ==========================================
# 1. 导入项目模块 (复用 src 中的定义)
# ==========================================
# 这样可以确保推理用的模型结构与训练时完全一致
from src.model import HerbTransformerGenerator
from src.config import CONFIG as TRAIN_CONFIG 

# ==========================================
# 2. 诊断与推理函数 (逻辑已更新)
# ==========================================
def batch_inference_debug(model_path, disease_folder, herb_matrix_file, output_file="final_results_debug.csv"):
    # 在这里定义推理专用的配置，强制使用 CPU 进行推理通常更稳妥，
    # 但参数 (d_model 等) 必须与训练时的 config.py 保持一致！
    INFERENCE_CONFIG = {
        'd_model': TRAIN_CONFIG['d_model'], 
        'nhead': TRAIN_CONFIG['nhead'], 
        'num_layers': TRAIN_CONFIG['num_layers'], 
        'max_seq_len': TRAIN_CONFIG['max_seq_len'],
        'device': 'cpu' # 推理时强制用 CPU，方便调试
    }
    
    print(">>> 1. Loading Knowledge Base (Herb Matrix)...")
    try:
        try: 
            df_herb = pd.read_csv(herb_matrix_file, index_col=0, encoding='utf-8')
        except: 
            df_herb = pd.read_csv(herb_matrix_file, index_col=0, encoding='gbk')
        
        df_herb.fillna(0, inplace=True)
        
        # [关键修复] 强制清理中药矩阵的索引（去除空格，转字符串）
        df_herb.index = df_herb.index.astype(str).str.strip()
        
        target_pathways = df_herb.index.tolist()
        herb_names = df_herb.columns.tolist()
        
        print(f"    Herb Matrix has {len(target_pathways)} pathways.")
        print(f"    Example pathways: {target_pathways[:3]}")
        
        herb_values = df_herb.T.values.astype(np.float32)
        # 归一化逻辑
        if herb_values.max() != herb_values.min():
            herb_values = 2 * (herb_values - herb_values.min()) / (herb_values.max() - herb_values.min()) - 1
        
        herb_nes_tensor = torch.tensor(herb_values).to(INFERENCE_CONFIG['device'])
        num_herbs, num_pathways = herb_nes_tensor.shape
        
    except Exception as e:
        print(f"Error loading herbs: {e}")
        return

    print("\n>>> 2. Loading Model...")
    # 初始化模型结构
    model = HerbTransformerGenerator(num_pathways, num_herbs, INFERENCE_CONFIG).to(INFERENCE_CONFIG['device'])
    
    # 加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=INFERENCE_CONFIG['device']))
        model.eval()
        print(f"    Model loaded successfully from {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        return
    
    # 搜索疾病文件
    files = glob.glob(os.path.join(disease_folder, "*.csv"))
    if not files: 
        files = glob.glob(os.path.join(disease_folder, "*.txt"))
    
    results = []
    
    print(f"\n>>> 3. Starting Inference for {len(files)} patients...\n")
    for file_path in files:
        file_name = os.path.basename(file_path)
        
        try:
            # --- 读取逻辑加强版 ---
            try: 
                df_d = pd.read_csv(file_path, encoding='utf-8')
            except: 
                df_d = pd.read_csv(file_path, encoding='gbk')
            
            # 清理列名
            df_d.columns = [str(c).strip() for c in df_d.columns]
            
            # 你的数据是两列，第一列是通路，第二列是NES
            # 我们直接按位置取，不依赖列名，这样更稳
            id_col = df_d.columns[0]
            nes_col = df_d.columns[1]
            
            # [关键修复] 清理疾病数据的通路名称（去除空格）
            df_d[id_col] = df_d[id_col].astype(str).str.strip()
            
            # 转换为 Series
            df_d = df_d.drop_duplicates(subset=[id_col]).set_index(id_col)[nes_col]
            
            # --- [诊断] 检查匹配率 ---
            # 看看疾病文件里的通路，有多少存在于中药矩阵里
            common_pathways = df_d.index.intersection(target_pathways)
            match_count = len(common_pathways)
            total_disease_pathways = len(df_d)
            
            if match_count == 0:
                print(f"🔴 WARNING: {file_name} has 0 matched pathways! (Total in file: {total_disease_pathways})")
                print(f"    File sample: {df_d.index.tolist()[:3]}")
                print(f"    Expected sample: {target_pathways[:3]}")
                results.append({"File": file_name, "Spearman_Corr": float('nan'), "Note": "No matching pathways"})
                continue
            
            if match_count < 5:
                 print(f"⚠️ LOW MATCH: {file_name} only has {match_count} matched pathways.")
            
            # 对齐数据
            base_s = pd.Series(0.0, index=target_pathways)
            # Reindex 会自动把缺失的填为 fill_value (0.0)
            aligned_d = df_d.reindex(base_s.index, fill_value=0.0).fillna(0.0)
            
            disease_vec = torch.tensor(aligned_d.values.astype(np.float32)).unsqueeze(0).to(INFERENCE_CONFIG['device'])
            
            # --- 推理 ---
            with torch.no_grad():
                current_input = torch.zeros(1, 1, INFERENCE_CONFIG['d_model']).to(INFERENCE_CONFIG['device'])
                logits_mask = torch.zeros(1, num_herbs).to(INFERENCE_CONFIG['device'])
                patient_formula = []
                total_effect = torch.zeros_like(disease_vec) 
                
                for t in range(INFERENCE_CONFIG['max_seq_len']):
                    herb_logits, dosage_pred, _ = model(disease_vec, tgt_seq=current_input)
                    herb_logits = herb_logits - (logits_mask * 1e9)
                    chosen_idx = torch.argmax(herb_logits, dim=-1)
                    dosage = dosage_pred.item()
                    
                    if dosage > 0.1:
                        herb_name = herb_names[chosen_idx.item()]
                        patient_formula.append(f"{herb_name}({dosage:.2f})")
                        one_hot = F.one_hot(chosen_idx, num_classes=num_herbs).float()
                        total_effect += torch.matmul(one_hot, herb_nes_tensor) * dosage

                    logits_mask.scatter_(1, chosen_idx.unsqueeze(1), 1.0)
                    next_embed = model.herb_embedding(chosen_idx) + model.dosage_projector(dosage_pred)
                    current_input = torch.cat([current_input, next_embed.unsqueeze(1)], dim=1)
            
            # --- 计算分数 ---
            # 转换为 numpy
            vec_disease = disease_vec.squeeze().cpu().numpy()
            vec_formula = total_effect.squeeze().cpu().numpy()
            
            # 只计算有效匹配的部分 (Mask: 疾病向量不为0的地方)
            mask = (np.abs(vec_disease) > 1e-6)
            
            if mask.sum() < 2:
                corr = float('nan') # 样本太少无法计算相关性
            else:
                # 计算预测的药效与疾病特征的相关性
                corr, _ = stats.spearmanr(vec_formula[mask], vec_disease[mask])
            
            results.append({
                "File": file_name,
                "Spearman_Corr": corr,
                "Matched_Pathways": match_count,
                "Formula": " + ".join(patient_formula)
            })
            
            print(f"✅ {file_name} | Match: {match_count}/{total_disease_pathways} | Spearman: {corr:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nDebug Inference Complete! Results saved to {output_file}")

if __name__ == "__main__":
    # 注意：这里的文件路径已根据之前整理的目录结构进行了调整
    batch_inference_debug(
        model_path="checkpoints/transformer_herb_generator_final.pth",
        disease_folder="data/disease_data_folder",
        herb_matrix_file="data/通路-中药组合_NES矩阵.csv"
    )