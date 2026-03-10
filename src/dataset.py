import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from .config import CONFIG
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ==========================================
# 辅助函数：处理单个文件
# (必须定义在类外面，否则多进程无法序列化)
# ==========================================
def process_single_file(file_path, target_pathways):
    try:
        # 尝试不同编码
        try:
            df_temp = pd.read_csv(file_path, encoding='utf-8')
        except:
            df_temp = pd.read_csv(file_path, encoding='gbk')

        # 清理列名空格
        df_temp.columns = [str(c).strip() for c in df_temp.columns]
        
        # 智能寻找 ID 和 NES 列
        id_col = None
        nes_col = None
        for col in df_temp.columns:
            col_upper = col.upper()
            if 'ID' in col_upper or '通路' in col_upper:
                id_col = col
            if 'NES' in col_upper or 'SCORE' in col_upper:
                nes_col = col
                
        # 找不到列名时的兜底
        if id_col is None or nes_col is None:
            df_temp = pd.read_csv(file_path, header=None)
            id_col = df_temp.columns[0]
            nes_col = df_temp.columns[1]

        # 去重并提取数据
        df_temp = df_temp.drop_duplicates(subset=[id_col])
        s_temp = df_temp.set_index(id_col)[nes_col]
        
        # 对齐数据（核心耗时步骤）
        # reindex 会自动填充 0.0，这步操作现在会在多个 CPU 核上同时进行
        aligned_s = s_temp.reindex(target_pathways, fill_value=0.0).fillna(0.0)
        
        return aligned_s.values.astype(np.float32)
        
    except Exception:
        # 出错返回 None，避免打印太多报错刷屏
        return None

# ==========================================
# 数据集类
# ==========================================
class RealDiseaseDataset(Dataset):
    def __init__(self, data_folder, target_pathways, augment=False):
        self.augment = augment
        self.data_matrix = []
        
        # 搜索文件
        file_list = glob.glob(os.path.join(data_folder, "*.csv"))
        if len(file_list) == 0:
             file_list = glob.glob(os.path.join(data_folder, "*.txt"))

        if len(file_list) == 0:
            print(f"Warning: No disease files found. Using dummy data.")
            self.data_matrix = np.random.randn(70, len(target_pathways)).astype(np.float32)
        else:
            # 动态决定使用的 CPU 核数 (最多用 16 核，防止系统卡死)
            cpu_count = multiprocessing.cpu_count()
            max_workers = min(cpu_count, 16)
            
            print(f"Found {len(file_list)} files. Loading with {max_workers} processes (Parallel)...")
            
            valid_results = []
            
            # === 多进程并行加载的核心代码 ===
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                # 注意：target_pathways 作为一个大列表传递给每个进程
                futures = [executor.submit(process_single_file, fp, target_pathways) for fp in file_list]
                
                # 使用 tqdm 显示并行进度
                for future in tqdm(futures, desc="Parallel Reading"):
                    res = future.result()
                    if res is not None:
                        valid_results.append(res)
            # ==============================

            if len(valid_results) > 0:
                self.data_matrix = np.stack(valid_results)
                print(f"Data Loaded. Shape: {self.data_matrix.shape}")
                print(f"Range: [{self.data_matrix.min():.2f}, {self.data_matrix.max():.2f}]")
            else:
                self.data_matrix = np.zeros((1, len(target_pathways)), dtype=np.float32)

    def __len__(self):
        if self.augment:
            return 3000 
        return len(self.data_matrix)

    def __getitem__(self, idx):
        if self.augment:
            real_idx = np.random.randint(0, len(self.data_matrix))
            sample = self.data_matrix[real_idx].copy()
            
            mask = (sample != 0)
            noise = np.random.normal(0, CONFIG['aug_noise'], sample.shape)
            sample[mask] = sample[mask] + noise[mask]
        else:
            sample = self.data_matrix[idx]
        return torch.tensor(sample, dtype=torch.float32)