import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats  

# ==========================================
# 1. 导入项目模块
# ==========================================
from src.model import HerbTransformerGenerator
from src.config import CONFIG as TRAIN_CONFIG 

# [核心修改] 导入所有 6 种可视化函数
from src.visualize import (
    generate_validation_plots, 
    plot_herb_pathway_heatmap, 
    plot_reversal_bubble_chart,
    plot_targeted_density,      # 新增
    plot_stepwise_trajectory,   # 新增
    plot_dosage_rose_chart,     # <--- 新增
    plot_herb_synergy_matrix,   # <--- 新增
    plot_reversal_radar_chart,  # <--- 新增
    plot_herb_pathway_network
)

# ==========================================
# 2. 诊断与推理函数
# ==========================================
def batch_inference_debug(model_path, disease_folder, herb_matrix_file, mapping_file, output_file="final_results_cn.csv"):
    INFERENCE_CONFIG = {
        'd_model': 256,       
        'nhead': 4,           
        'num_layers': 2,      
        'max_seq_len': 20,    
        'device': 'cpu'       
    }
    
    print(f">>> 1. Loading Mapping from {mapping_file}...")
    try:
        df_map = pd.read_excel(mapping_file)
        map_names = df_map.iloc[:, 0].astype(str).str.strip()
        map_ids = df_map.iloc[:, 1].astype(str).str.strip()
        id_to_name = dict(zip(map_ids, map_names))
        print(f"    Loaded {len(id_to_name)} herbs into dictionary.")
    except Exception as e:
        print(f"❌ Error loading mapping file: {e}")
        return

    print("\n>>> 2. Loading Knowledge Base...")
    try:
        try: 
            df_herb = pd.read_csv(herb_matrix_file, index_col=0, encoding='utf-8')
        except: 
            df_herb = pd.read_csv(herb_matrix_file, index_col=0, encoding='gbk')
        
        df_herb.fillna(0, inplace=True)
        raw_herb_names = df_herb.columns.tolist()
        herb_names = [str(x).strip() for x in raw_herb_names]

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

    print("\n>>> 3. Loading Model...")
    model = HerbTransformerGenerator(num_pathways, num_herbs, INFERENCE_CONFIG).to(INFERENCE_CONFIG['device'])
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=INFERENCE_CONFIG['device'], weights_only=True))
        model.eval()
    else:
        print(f"❌ Model file not found.")
        return
    
    files = glob.glob(os.path.join(disease_folder, "*.csv"))
    if not files: files = glob.glob(os.path.join(disease_folder, "*.txt"))
    
    results = []
    print(f"\n>>> 4. Starting Inference for {len(files)} patients...\n")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
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
            
            # --- [为可视化提取疾病最核心的前50条通路] ---
            vec_disease_np_full = disease_vec.squeeze().cpu().numpy()
            top_pathway_indices = np.argsort(np.abs(vec_disease_np_full))[-50:]
            # ----------------------------------------
            
            with torch.no_grad():
                current_input = torch.zeros(1, 1, INFERENCE_CONFIG['d_model']).to(INFERENCE_CONFIG['device'])
                logits_mask = torch.zeros(1, num_herbs).to(INFERENCE_CONFIG['device'])
                
                patient_formula_ids = []  
                all_chinese_names_flat = [] 
                
                # 收集可视化所需数据的容器
                dosages_list = []
                selected_herbs_nes = []
                selected_herbs_names_for_plot = []
                stepwise_corrs = [] # [新增] 存储每一步的相关性
                
                total_effect = torch.zeros_like(disease_vec) 
                
                vec_disease_np = disease_vec.squeeze().cpu().numpy()
                mask_disease = (np.abs(vec_disease_np) > 1e-6)

                for t in range(INFERENCE_CONFIG['max_seq_len']):
                    herb_logits, dosage_pred, _ = model(disease_vec, tgt_seq=current_input)
                    herb_logits = herb_logits - (logits_mask * 1e9)
                    chosen_idx = torch.argmax(herb_logits, dim=-1)
                    dosage = dosage_pred.item()
                    
                    if dosage > 0.1:
                        raw_id_str = herb_names[chosen_idx.item()]
                        patient_formula_ids.append(f"{raw_id_str}({dosage:.2f})")
                        
                        parts = raw_id_str.split('+') 
                        herb_display_name = []
                        for part in parts:
                            clean_part = part.strip()
                            c_name = id_to_name.get(clean_part, clean_part)
                            all_chinese_names_flat.append(c_name)
                            herb_display_name.append(c_name)
                            
                        # --- [收集作图数据] ---
                        dosages_list.append(dosage)
                        selected_herbs_names_for_plot.append("+".join(herb_display_name))
                        
                        # 截取该药物在这 Top50 核心通路上的药效特征
                        herb_nes_full = herb_nes_tensor[chosen_idx].squeeze().cpu().numpy()
                        selected_herbs_nes.append(herb_nes_full[top_pathway_indices])
                        # -----------------------
                        
                        one_hot = F.one_hot(chosen_idx, num_classes=num_herbs).float()
                        total_effect += torch.matmul(one_hot, herb_nes_tensor) * dosage
                        
                        # [新增核心逻辑] 计算当前步骤的累积疗效相关性
                        vec_current_formula = total_effect.squeeze().cpu().numpy()
                        if mask_disease.sum() >= 2:
                            curr_corr, _ = stats.spearmanr(vec_current_formula[mask_disease], vec_disease_np[mask_disease])
                            if not np.isnan(curr_corr):
                                stepwise_corrs.append(curr_corr)

                    logits_mask.scatter_(1, chosen_idx.unsqueeze(1), 1.0)
                    next_embed = model.herb_embedding(chosen_idx) + model.dosage_projector(dosage_pred)
                    current_input = torch.cat([current_input, next_embed.unsqueeze(1)], dim=1)
            
            # 计算最终相关性
            vec_formula = total_effect.squeeze().cpu().numpy()
            corr = float('nan')
            if mask_disease.sum() >= 2:
                corr, _ = stats.spearmanr(vec_formula[mask_disease], vec_disease_np[mask_disease])
            
            # ==========================================
            # [核心修改]: 在此处一键调用所有 6 种可视化代码！
            # ==========================================
            if not np.isnan(corr):
                # 1. 生成全局验证图 (散点图、蝴蝶图、随机对抗小提琴图)
                generate_validation_plots(
                    vec_disease_np, vec_formula, corr, file_name, 
                    result_folder="validation_results",
                    herb_matrix=herb_nes_tensor.cpu().numpy(),
                    dosages=dosages_list
                )
                
                # 2. 生成 Top 20 靶向逆转特色气泡图
                try:
                    plot_reversal_bubble_chart(
                        vec_disease=vec_disease_np,
                        vec_formula=vec_formula,
                        file_name=file_name,
                        result_folder="validation_results"
                    )
                except Exception as e:
                    print(f"❌ 气泡图调用失败: {e}")

                # 3. [新增] 生成靶向干预密度分布图 (机制验证核心)
                try:
                    plot_targeted_density(
                        vec_disease=vec_disease_np,
                        vec_formula=vec_formula,
                        file_name=file_name,
                        result_folder="validation_results"
                    )
                except Exception as e:
                    print(f"❌ 密度图调用失败: {e}")

                # 4. [新增] 生成君臣佐使效用累积轨迹图 (AI生成过程)
                try:
                    plot_stepwise_trajectory(
                        stepwise_corrs=stepwise_corrs,
                        file_name=file_name,
                        result_folder="validation_results"
                    )
                except Exception as e:
                    print(f"❌ 轨迹图调用失败: {e}")
                
                if len(selected_herbs_names_for_plot) > 0:
                    # 5. 生成中药-通路干预热力图 (机制解释图)
                    plot_herb_pathway_heatmap(
                        herb_names=selected_herbs_names_for_plot,
                        herb_nes_matrix=np.array(selected_herbs_nes),
                        file_name=file_name,
                        result_folder="validation_results"
                    )
                    # 6. [新增] 绘制君臣佐使南丁格尔玫瑰图
                    try:
                        plot_dosage_rose_chart(
                            herb_names=selected_herbs_names_for_plot,
                            dosages=dosages_list,
                            file_name=file_name,
                            result_folder="validation_results"
                        )
                    except Exception as e:
                        print(f"❌ 玫瑰图调用失败: {e}")

                    # 7. [新增] 绘制中药协同-互补矩阵图
                    try:
                        plot_herb_synergy_matrix(
                            herb_names=selected_herbs_names_for_plot,
                            herb_nes_matrix=np.array(selected_herbs_nes),
                            file_name=file_name,
                            result_folder="validation_results"
                        )
                    except Exception as e:
                        print(f"❌ 协同矩阵调用失败: {e}")
                    # 8. [新增] 绘制多维靶向逆转雷达图
                    try:
                        plot_reversal_radar_chart(
                            vec_disease=vec_disease_np,
                            vec_formula=vec_formula,
                            file_name=file_name,
                            result_folder="validation_results"
                        )
                    except Exception as e:
                        print(f"❌ 雷达图调用失败: {e}")

                    if len(selected_herbs_names_for_plot) > 0:
                        # 9. [新增] 绘制核心药理靶向网络图
                        try:
                            plot_herb_pathway_network(
                                herb_names=selected_herbs_names_for_plot,
                                herb_nes_matrix=np.array(selected_herbs_nes),
                                file_name=file_name,
                                result_folder="validation_results"
                            )
                        except Exception as e:
                            print(f"❌ 网络图调用失败: {e}")
            # ==========================================

            unique_chinese_names = list(dict.fromkeys(all_chinese_names_flat))
            
            results.append({
                "File": file_name,
                "Spearman_Corr": corr,
                "Matched_Pathways": match_count,
                "Formula_ID_Detail": " + ".join(patient_formula_ids), 
                "Chinese_Formula": " + ".join(unique_chinese_names)   
            })
            
            print(f"✅ {file_name} | Spearman: {corr:.4f} | Visualizations saved.")
            
        except Exception as e:
            import traceback
            print(f"❌ Error processing {file_name}: {e}")
            print(traceback.format_exc())

    df_res = pd.DataFrame(results)
    target_cols = ["File", "Spearman_Corr", "Matched_Pathways", "Formula_ID_Detail", "Chinese_Formula"]
    final_order = [c for c in target_cols if c in df_res.columns] + \
                  [c for c in df_res.columns if c not in target_cols]
    df_res = df_res[final_order]
    
    df_res.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nDone! Results in {output_file}")

if __name__ == "__main__":
    batch_inference_debug(
        model_path="checkpoints/transformer_herb_generator_nozero.pth",
        disease_folder="data/disease_human",
        herb_matrix_file="data/通路-中药组合_NES矩阵.csv",
        mapping_file="data/核心中药20251112-V4.xlsx"
    )