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

def batch_inference_debug(model_path, disease_folder, herb_matrix_file, mapping_file, output_file="final_results_cn.csv"):
    INFERENCE_CONFIG = {
        'd_model': 256,       
        'nhead': 4,           
        'num_layers': 2,      
        'max_seq_len': 20,    
        'device': 'cpu'       
    }
    
    # ==========================================
    # 1. 加载映射表 (ID -> 中文名)
    # ==========================================
    print(f">>> 1. Loading Mapping from {mapping_file}...")
    try:
        df_map = pd.read_excel(mapping_file)
        
        # 强制取第1列(中文)和第2列(ID)，并去空格
        map_names = df_map.iloc[:, 0].astype(str).str.strip()
        map_ids = df_map.iloc[:, 1].astype(str).str.strip()
        
        # 建立字典: 单个ID -> 中文名
        id_to_name = dict(zip(map_ids, map_names))
        
        print(f"    Loaded {len(id_to_name)} herbs into dictionary.")
        
    except Exception as e:
        print(f"❌ Error loading mapping file: {e}")
        return

    # ==========================================
    # 2. 加载知识库 (获取模型的列名)
    # ==========================================
    print("\n>>> 2. Loading Knowledge Base...")
    try:
        try: 
            df_herb = pd.read_csv(herb_matrix_file, index_col=0, encoding='utf-8')
        except: 
            df_herb = pd.read_csv(herb_matrix_file, index_col=0, encoding='gbk')
        
        df_herb.fillna(0, inplace=True)
        # 获取列名，这可能是组合ID，例如 "HERB_A" 或 "HERB_A+HERB_B"
        raw_herb_names = df_herb.columns.tolist()
        herb_names = [str(x).strip() for x in raw_herb_names]
        
        # --- [DEBUG] 测试拆分逻辑 ---
        print("\n    🛑 [DEBUG] Testing Split Logic on first model ID...")
        test_id = herb_names[0] # 假设是 "HERB_1020+HERB_1332"
        print(f"    Raw Model ID: {test_id}")
        parts = test_id.split('+')
        print(f"    Split parts: {parts}")
        mapped_test = [id_to_name.get(p.strip(), p.strip()) for p in parts]
        print(f"    Mapped result: {mapped_test}")
        # ----------------------------

        df_herb.index = df_herb.index.astype(str).str.strip()
        target_pathways = df_herb.index.tolist()
        
        herb_values = df_herb.T.values.astype(np.float32)
        if herb_values.max() != herb_values.min():
            herb_values = 2 * (herb_values - herb_values.min()) / (herb_values.max() - herb_values.min()) - 1
        
        herb_nes_tensor = torch.tensor(herb_values).to(INFERENCE_CONFIG['device'])
        num_herbs, num_pathways = herb_nes_tensor.shape
        
    except Exception as e:
        print(f"Error loading herbs: {e}")
        return

    # ==========================================
    # 3. 加载模型
    # ==========================================
    print("\n>>> 3. Loading Model...")
    model = HerbTransformerGenerator(num_pathways, num_herbs, INFERENCE_CONFIG).to(INFERENCE_CONFIG['device'])
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=INFERENCE_CONFIG['device']))
        model.eval()
    else:
        print(f"❌ Model file not found.")
        return
    
    # ==========================================
    # 4. 推理循环
    # ==========================================
    files = glob.glob(os.path.join(disease_folder, "*.csv"))
    if not files: files = glob.glob(os.path.join(disease_folder, "*.txt"))
    
    results = []
    print(f"\n>>> 4. Starting Inference for {len(files)} patients...\n")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            # 读取数据
            try: df_d = pd.read_csv(file_path, encoding='utf-8')
            except: df_d = pd.read_csv(file_path, encoding='gbk')
            
            df_d.columns = [str(c).strip() for c in df_d.columns]
            id_col_d = df_d.columns[0]
            nes_col_d = df_d.columns[1]
            df_d[id_col_d] = df_d[id_col_d].astype(str).str.strip()
            df_d = df_d.drop_duplicates(subset=[id_col_d]).set_index(id_col_d)[nes_col_d]
            
            match_count = len(df_d.index.intersection(target_pathways))
            
            if match_count == 0:
                results.append({"File": file_name, "Spearman_Corr": float('nan')})
                continue
            
            base_s = pd.Series(0.0, index=target_pathways)
            aligned_d = df_d.reindex(base_s.index, fill_value=0.0).fillna(0.0)
            disease_vec = torch.tensor(aligned_d.values.astype(np.float32)).unsqueeze(0).to(INFERENCE_CONFIG['device'])
            
            with torch.no_grad():
                current_input = torch.zeros(1, 1, INFERENCE_CONFIG['d_model']).to(INFERENCE_CONFIG['device'])
                logits_mask = torch.zeros(1, num_herbs).to(INFERENCE_CONFIG['device'])
                
                patient_formula_ids = []  # 存：HERB_A+HERB_B(0.9)
                all_chinese_names_flat = [] # 扁平化列表，用于存拆分后的所有中文名
                
                total_effect = torch.zeros_like(disease_vec) 
                
                for t in range(INFERENCE_CONFIG['max_seq_len']):
                    herb_logits, dosage_pred, _ = model(disease_vec, tgt_seq=current_input)
                    herb_logits = herb_logits - (logits_mask * 1e9)
                    chosen_idx = torch.argmax(herb_logits, dim=-1)
                    dosage = dosage_pred.item()
                    
                    if dosage > 0.1:
                        # 1. 获取原始组合ID (例如 "HERB_1020+HERB_1332")
                        raw_id_str = herb_names[chosen_idx.item()]
                        
                        # 2. 存入第4列用的列表 (保留原始结构和剂量)
                        patient_formula_ids.append(f"{raw_id_str}({dosage:.2f})")
                        
                        # 3. [核心修改] 拆分 + 查表 + 存入第5列列表
                        # 无论有没有 '+', split都能正常工作
                        parts = raw_id_str.split('+') 
                        
                        for part in parts:
                            clean_part = part.strip()
                            # 查表，找不到就返回ID本身
                            c_name = id_to_name.get(clean_part, clean_part)
                            all_chinese_names_flat.append(c_name)
                        
                        one_hot = F.one_hot(chosen_idx, num_classes=num_herbs).float()
                        total_effect += torch.matmul(one_hot, herb_nes_tensor) * dosage

                    logits_mask.scatter_(1, chosen_idx.unsqueeze(1), 1.0)
                    next_embed = model.herb_embedding(chosen_idx) + model.dosage_projector(dosage_pred)
                    current_input = torch.cat([current_input, next_embed.unsqueeze(1)], dim=1)
            
            # 计算相关性
            vec_disease = disease_vec.squeeze().cpu().numpy()
            vec_formula = total_effect.squeeze().cpu().numpy()
            mask = (np.abs(vec_disease) > 1e-6)
            corr = float('nan')
            if mask.sum() >= 2:
                corr, _ = stats.spearmanr(vec_formula[mask], vec_disease[mask])
            
            # --- [去重逻辑] ---
            # 使用 dict.fromkeys 保留 "all_chinese_names_flat" 出现的顺序 (即按照剂量由高到低的顺序)
            unique_chinese_names = list(dict.fromkeys(all_chinese_names_flat))
            
            results.append({
                "File": file_name,
                "Spearman_Corr": corr,
                "Matched_Pathways": match_count,
                "Formula_ID_Detail": " + ".join(patient_formula_ids), # 第4列
                "Chinese_Formula": " + ".join(unique_chinese_names)   # 第5列 (拆分翻译后去重)
            })
            
            print(f"✅ {file_name} | Spearman: {corr:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

    # ==========================================
    # 5. 保存结果，强制列顺序
    # ==========================================
    df_res = pd.DataFrame(results)
    
    # 强制指定顺序
    # 1: File, 2: Spearman_Corr, 3: Matched_Pathways, 4: Formula_ID_Detail, 5: Chinese_Formula
    target_cols = ["File", "Spearman_Corr", "Matched_Pathways", "Formula_ID_Detail", "Chinese_Formula"]
    
    # 确保只选存在的列，并把其余列放到后面
    final_order = [c for c in target_cols if c in df_res.columns] + \
                  [c for c in df_res.columns if c not in target_cols]
    
    df_res = df_res[final_order]
    
    df_res.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nDone! Results in {output_file}")
    print(f"Check Column 5 (Chinese_Formula) for translated names.")

if __name__ == "__main__":
    batch_inference_debug(
        model_path="checkpoints/transformer_herb_generator_nozero.pth",
        disease_folder="data/disease_human",
        herb_matrix_file="data/通路-中药组合_NES矩阵.csv",
        mapping_file="data/核心中药20251112-V4.xlsx"
    )