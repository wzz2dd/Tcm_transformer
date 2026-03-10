# 文件路径: src/visualize.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr  # [新增] 用于计算随机组合的相关性

# 移除中文字体设置，防止 Linux 报错
plt.rcParams['axes.unicode_minus'] = False 

def generate_validation_plots(vec_disease, vec_formula, spearman_corr, file_name, 
                              result_folder="validation_results", 
                              herb_matrix=None, dosages=None):
    """
    生成疾病与方剂通路特征的对比验证图 
    (散点图、双向柱状图、小提琴对比图、逆转统计)
    """
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')
    
    # 过滤掉疾病评分为 0 的无效通路（只看患病发生改变的通路）
    mask = (vec_disease != 0)
    test = vec_disease[mask]
    sum_adjusted = vec_formula[mask]
    
    n_pathways = len(test)
    if n_pathways < 2: 
        return # 数据过少则跳过
    
    # ==========================================
    # 核心优化：使用 MaxAbs 缩放，保留相对大小且0点对齐
    # ==========================================
    max_test = np.max(np.abs(test)) if np.max(np.abs(test)) != 0 else 1
    max_form = np.max(np.abs(sum_adjusted)) if np.max(np.abs(sum_adjusted)) != 0 else 1
    
    test_scaled = test / max_test
    form_scaled = sum_adjusted / max_form

    # ==========================================
    # 1. 绘制 Spearman 负相关散点图 (全局趋势)
    # ==========================================
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.regplot(x=form_scaled, y=test_scaled, 
                scatter_kws={'alpha':0.5, 'color': '#4285F4'}, 
                line_kws={'color': '#FBBC05', 'linewidth': 2.5})
    plt.title(f'Spearman Correlation: {spearman_corr:.4f}\n(N={n_pathways} Active Pathways)', fontsize=15)
    plt.xlabel('Formula Intervention Score (Relative)', fontsize=13)
    plt.ylabel('Disease Perturbation Score (Relative)', fontsize=13)
    plt.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'{result_folder}/{plot_name}_scatter.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # ==========================================
    # 2. 绘制 Diverging Bar Chart (蝴蝶图)
    # ==========================================
    MAX_DISPLAY = 100  
    
    if n_pathways > MAX_DISPLAY:
        top_idx = np.argsort(np.abs(test_scaled))[-MAX_DISPLAY:]
        test_plot = test_scaled[top_idx]
        form_plot = form_scaled[top_idx]
        plot_n = MAX_DISPLAY
        plot_title_suffix = f"(Top {MAX_DISPLAY} core pathways shown)"
    else:
        test_plot = test_scaled
        form_plot = form_scaled
        plot_n = n_pathways
        plot_title_suffix = f"(N={n_pathways})"

    sort_idx = np.argsort(test_plot)
    sorted_test = test_plot[sort_idx]
    sorted_form = form_plot[sort_idx]

    fig_height = max(4.0, plot_n * 0.25)
    plt.figure(figsize=(8, fig_height))
    sns.set_theme(style="white", font_scale=1.1)
    
    y_pos = np.arange(plot_n)
    plt.barh(y_pos, sorted_test, height=0.6, color='#EA4335', alpha=0.8, label='Disease Pathway (Target)')
    plt.barh(y_pos, sorted_form, height=0.6, color='#34A853', alpha=0.8, label='Formula Intervention')
    plt.axvline(0, color='black', linewidth=1.2)
    plt.yticks(y_pos, [f"Pathway {i+1}" for i in range(plot_n)], fontsize=9)
    plt.xlabel('Relative NES Score', fontsize=12)
    plt.title(f'Disease vs. Formula Pathway Reversal Profile\n{plot_title_suffix}', fontsize=14, pad=15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    sns.despine(left=True, bottom=True)
    plt.savefig(f"{result_folder}/{plot_name}_diverging_bar.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # ==========================================
    # 3. [新增] 绘制模型置信度对比小提琴图 (对抗随机性验证)
    # ==========================================
    if herb_matrix is not None and dosages is not None and len(dosages) > 0:
        np.random.seed(42) # 固定随机种子以保证结果可复现
        random_corrs = []
        num_total_herbs = herb_matrix.shape[0]
        k_herbs = len(dosages)
        dosages_arr = np.array(dosages)
        
        # 仅截取对疾病有效的通路进行计算，加速运算
        herb_matrix_masked = herb_matrix[:, mask]
        
        # 进行 1000 次蒙特卡洛随机模拟
        for _ in range(1000):
            # 随机抽取同等数量(K)的中药
            rand_indices = np.random.choice(num_total_herbs, k_herbs, replace=False)
            rand_herbs_features = herb_matrix_masked[rand_indices, :]
            
            # 使用模型优化出的同样剂量进行叠加
            rand_formula_effect = np.dot(dosages_arr, rand_herbs_features)
            
            # 计算随机配方的 Spearman 相关性
            r_corr, _ = spearmanr(rand_formula_effect, test)
            if not np.isnan(r_corr):
                random_corrs.append(r_corr)
                
        random_corrs = np.array(random_corrs)
        
        # 计算经验 P 值 (Empirical P-value)
        # 意义：在 1000 个随机配方中，有多少个配方的抗病效果（负相关性）比 AI 生成的还要好？
        p_val = np.sum(random_corrs <= spearman_corr) / len(random_corrs)
        
        # 开始绘图
        plt.figure(figsize=(6, 6))
        sns.set_theme(style="whitegrid", font_scale=1.2)
        
        # 绘制随机分布的小提琴图
        sns.violinplot(y=random_corrs, color="lightgray", inner="quartile", linewidth=1.5)
        
        # 把我们 AI 预测的分数打在图上 (红色的星星)
        plt.scatter(x=0, y=spearman_corr, color='#EA4335', s=300, marker='*', zorder=10, edgecolor='black', 
                    label=f'AI Formula\n(Score: {spearman_corr:.4f})')
        
        plt.title(f'AI Model vs. 1000 Random Formulas\nEmpirical P-value = {p_val:.4f}', fontsize=14, pad=15)
        plt.ylabel('Spearman Correlation (More negative is better)', fontsize=12)
        plt.xticks([]) # 隐藏X轴
        plt.legend(loc='upper right', frameon=True, shadow=True)
        
        plt.savefig(f"{result_folder}/{plot_name}_violin.pdf", format="pdf", bbox_inches="tight")
        plt.close()

    # ==========================================
    # 4. 靶向逆转匹配度统计
    # ==========================================
    dis_pos = set(np.where(test > 0)[0])
    dis_neg = set(np.where(test < 0)[0])
    for_pos = set(np.where(sum_adjusted > 0)[0])
    for_neg = set(np.where(sum_adjusted < 0)[0])
    
    reversal_count = len(dis_pos & for_neg) + len(dis_neg & for_pos)
    total_active = len(dis_pos) + len(dis_neg)
    
    with open(f"{result_folder}/reversal_stats.txt", "a", encoding="utf-8") as f:
        f.write(f"[{plot_name}] Reversal Pathways: {reversal_count}/{total_active} ({(reversal_count/total_active)*100:.2f}%)\n")