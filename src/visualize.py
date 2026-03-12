# 文件路径: src/visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties, fontManager

# ==========================================
# [终极修复]：强制加载本地字体 + 拦截 Seaborn 霸道重置
# ==========================================
font_path = "simhei.ttf"  # 确保这个文件在你的根目录下！

if os.path.exists(font_path):
    # 将本地字体强行加入 matplotlib 字体库
    fontManager.addfont(font_path)
    custom_font = FontProperties(fname=font_path).get_name()
    print(f"✅ 成功加载本地中文字体: {custom_font}")
else:
    print(f"⚠️ 警告: 未在根目录找到 {font_path}，图表中的中文将显示为方块！")
    custom_font = "sans-serif"

# 拦截并重写 seaborn 的 set_theme 函数，防止它把我们的中文字体洗掉
_original_set_theme = sns.set_theme
def _safe_set_theme(*args, **kwargs):
    _original_set_theme(*args, **kwargs)
    plt.rcParams['font.sans-serif'] = [custom_font, 'SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
sns.set_theme = _safe_set_theme

# 初始化一次全局配置
plt.rcParams['font.sans-serif'] = [custom_font, 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 散点图 + 2. 蝴蝶图 + 3. 小提琴图
# ==========================================
def generate_validation_plots(vec_disease, vec_formula, spearman_corr, file_name, 
                              result_folder="validation_results", 
                              herb_matrix=None, dosages=None):
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')
    
    mask = (vec_disease != 0)
    test = vec_disease[mask]
    sum_adjusted = vec_formula[mask]
    
    n_pathways = len(test)
    if n_pathways < 2: return
    
    max_test = np.max(np.abs(test)) if np.max(np.abs(test)) != 0 else 1
    max_form = np.max(np.abs(sum_adjusted)) if np.max(np.abs(sum_adjusted)) != 0 else 1
    test_scaled = test / max_test
    form_scaled = sum_adjusted / max_form

    # --- 1. 散点图 ---
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.regplot(x=form_scaled, y=test_scaled, scatter_kws={'alpha':0.5, 'color': '#4285F4'}, line_kws={'color': '#FBBC05', 'linewidth': 2.5})
    plt.title(f'Spearman Correlation: {spearman_corr:.4f}\n(N={n_pathways} Active Pathways)', fontsize=15)
    plt.xlabel('Formula Intervention Score (Relative)', fontsize=13)
    plt.ylabel('Disease Perturbation Score (Relative)', fontsize=13)
    plt.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    plt.savefig(f'{result_folder}/{plot_name}_scatter.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # --- 2. 蝴蝶图 ---
    MAX_DISPLAY = 100  
    if n_pathways > MAX_DISPLAY:
        top_idx = np.argsort(np.abs(test_scaled))[-MAX_DISPLAY:]
        test_plot = test_scaled[top_idx]
        form_plot = form_scaled[top_idx]
        plot_n = MAX_DISPLAY
        plot_title_suffix = f"(Top {MAX_DISPLAY} core pathways)"
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
    plt.yticks(y_pos, [f"P_{i+1}" for i in range(plot_n)], fontsize=9)
    plt.xlabel('Relative NES Score', fontsize=12)
    plt.title(f'Disease vs. Formula Pathway Reversal\n{plot_title_suffix}', fontsize=14, pad=15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    sns.despine(left=True, bottom=True)
    plt.savefig(f"{result_folder}/{plot_name}_diverging_bar.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    # --- 3. 小提琴图 ---
    if herb_matrix is not None and dosages is not None and len(dosages) > 0:
        np.random.seed(42)
        random_corrs = []
        num_total_herbs = herb_matrix.shape[0]
        k_herbs = len(dosages)
        dosages_arr = np.array(dosages)
        herb_matrix_masked = herb_matrix[:, mask]
        
        for _ in range(1000):
            rand_indices = np.random.choice(num_total_herbs, k_herbs, replace=False)
            rand_formula_effect = np.dot(dosages_arr, herb_matrix_masked[rand_indices, :])
            r_corr, _ = spearmanr(rand_formula_effect, test)
            if not np.isnan(r_corr):
                random_corrs.append(r_corr)
                
        random_corrs = np.array(random_corrs)
        p_val = np.sum(random_corrs <= spearman_corr) / len(random_corrs) if len(random_corrs) > 0 else 0.0
        
        plt.figure(figsize=(6, 6))
        sns.set_theme(style="whitegrid", font_scale=1.2)
        sns.violinplot(y=random_corrs, color="lightgray", inner="quartile", linewidth=1.5)
        plt.scatter(x=0, y=spearman_corr, color='#EA4335', s=300, marker='*', zorder=10, edgecolor='black', label=f'AI Formula\n(Score: {spearman_corr:.4f})')
        plt.title(f'AI Model vs. 1000 Random Formulas\nEmpirical P-value = {p_val:.4f}', fontsize=14, pad=15)
        plt.ylabel('Spearman Correlation (More negative is better)', fontsize=12)
        plt.xticks([])
        plt.legend(loc='upper right', frameon=True, shadow=True)
        plt.savefig(f"{result_folder}/{plot_name}_violin.pdf", format="pdf", bbox_inches="tight")
        plt.close()

# ==========================================
# 4. 中药-通路热力图
# ==========================================
def plot_herb_pathway_heatmap(herb_names, herb_nes_matrix, file_name, result_folder="validation_results"):
    if len(herb_names) == 0 or herb_nes_matrix.shape[0] == 0: return
    
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')
    
    plt.figure(figsize=(16, max(5, len(herb_names) * 0.8)))
    sns.set_theme(style="white", font_scale=1.0)
    
    ax = sns.heatmap(herb_nes_matrix, cmap='coolwarm', center=0, 
                xticklabels=[f"Pathway {i+1}" for i in range(herb_nes_matrix.shape[1])], 
                yticklabels=herb_names, 
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'NES Value', 'shrink': 0.8})
    
    plt.title(f"Herb-Pathway Interaction Profile\n({plot_name})", fontsize=16, pad=15)
    plt.xlabel("Top Core Biological Pathways", fontsize=14)
    plt.ylabel("Selected Herbs (中药配伍)", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{result_folder}/{plot_name}_herb_path_heatmap.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 5. 靶向逆转气泡图
# ==========================================
def plot_reversal_bubble_chart(vec_disease, vec_formula, file_name, result_folder="validation_results"):
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')
    
    mask = (vec_disease != 0)
    test = vec_disease[mask]
    sum_adjusted = vec_formula[mask]
    
    if len(test) < 2: return
    
    top_n = min(20, len(test))
    top_idx = np.argsort(np.abs(test))[-top_n:]
    
    disease_scores = test[top_idx]
    formula_scores = sum_adjusted[top_idx]
    
    sort_order = np.argsort(disease_scores)
    disease_scores = disease_scores[sort_order]
    formula_scores = formula_scores[sort_order]
    pathway_names = [f"Pathway {i+1}" for i in range(top_n)] 
    
    plt.figure(figsize=(8, max(5, top_n * 0.4)))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    max_abs_score = np.max(np.abs(disease_scores)) if np.max(np.abs(disease_scores)) != 0 else 1
    bubble_sizes = (np.abs(disease_scores) / max_abs_score) * 1000 + 100 
    
    y_positions = np.arange(top_n)
    scatter = plt.scatter(x=formula_scores, y=y_positions, 
                          s=bubble_sizes, 
                          c=disease_scores, cmap='coolwarm', 
                          alpha=0.8, edgecolors='white', linewidth=1.5)
    plt.yticks(y_positions, pathway_names)
    
    plt.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.5)
    
    cbar = plt.colorbar(scatter, shrink=0.5, pad=0.02)
    cbar.set_label('Disease Baseline NES\n(Red=Up, Blue=Down)', fontsize=11)
    
    plt.title(f"Top {top_n} Core Pathways Reversal Bubble Chart\n({plot_name})", fontsize=14, pad=15)
    plt.xlabel("Formula Intervention Score (NES)", fontsize=13)
    plt.ylabel("Core Biological Pathways", fontsize=13)
    
    sns.despine(left=True, bottom=True)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{result_folder}/{plot_name}_bubble_chart.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 6. 靶向干预密度分布图
# ==========================================
def plot_targeted_density(vec_disease, vec_formula, file_name, result_folder="validation_results"):
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')
    
    mask = (vec_disease != 0)
    disease_masked = vec_disease[mask]
    formula_masked = vec_formula[mask]
    
    up_indices = disease_masked > 0
    down_indices = disease_masked < 0
    formula_on_up = formula_masked[up_indices]
    formula_on_down = formula_masked[down_indices]
    
    if len(formula_on_up) < 2 or len(formula_on_down) < 2: return

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    sns.kdeplot(formula_on_up, color='#EA4335', fill=True, alpha=0.3, linewidth=2.5, label='Formula Effect on\nDisease UP-regulated Pathways')
    sns.kdeplot(formula_on_down, color='#4285F4', fill=True, alpha=0.3, linewidth=2.5, label='Formula Effect on\nDisease DOWN-regulated Pathways')
    
    plt.axvline(0, color='grey', linestyle='--', linewidth=1.5)
    
    plt.title(f"Targeted Intervention Density Profile\n({plot_name})", fontsize=15, pad=15)
    plt.xlabel("Formula Intervention Score (NES)\n(<0: Inhibitory, >0: Activating)", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False)
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(f"{result_folder}/{plot_name}_density_plot.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 7. 君臣佐使效用累积轨迹图
# ==========================================
def plot_stepwise_trajectory(stepwise_corrs, file_name, result_folder="validation_results"):
    if len(stepwise_corrs) < 2: return
    
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')
    
    steps = np.arange(1, len(stepwise_corrs) + 1)
    
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    plt.plot(steps, stepwise_corrs, marker='o', linestyle='-', linewidth=2.5, color='#34A853', markersize=8, label='Cumulative Efficacy')
    plt.scatter(steps[0], stepwise_corrs[0], color='#EA4335', s=150, zorder=5, label='Start (Jun Herb)')
    plt.scatter(steps[-1], stepwise_corrs[-1], color='#4285F4', s=150, zorder=5, label='End (Final Formula)')

    plt.title(f"Step-wise Efficacy Accumulation Trajectory\n({plot_name})", fontsize=15, pad=15)
    plt.xlabel("Generation Step (Herb Sequence)", fontsize=13)
    plt.ylabel("Spearman Correlation with Disease\n(Lower is Better)", fontsize=13)
    plt.xticks(steps)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{result_folder}/{plot_name}_trajectory_plot.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 8. 君臣佐使剂量南丁格尔玫瑰图
# ==========================================
def plot_dosage_rose_chart(herb_names, dosages, file_name, result_folder="validation_results"):
    if len(herb_names) < 2: return
    
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')

    sort_idx = np.argsort(dosages)[::-1]
    sorted_herbs = [herb_names[i] for i in sort_idx]
    sorted_dosages = [dosages[i] for i in sort_idx]

    N = len(sorted_herbs)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    colors = cm.coolwarm(np.linspace(0.9, 0.1, N))

    bars = ax.bar(angles, sorted_dosages, width=width, bottom=max(sorted_dosages)*0.2, 
                  color=colors, alpha=0.85, edgecolor='white', linewidth=2.5)

    ax.set_xticks(angles)
    ax.set_xticklabels(sorted_herbs, fontsize=12, fontweight='bold')
    ax.set_yticklabels([]) 
    ax.spines['polar'].set_visible(False) 
    ax.grid(color='#EEEEEE', linestyle='--', linewidth=1.5)

    plt.title(f"AI Formula Dosage Hierarchy (君臣佐使权重)\n{plot_name}", fontsize=16, pad=30)
    plt.tight_layout()
    plt.savefig(f"{result_folder}/{plot_name}_rose_chart.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# 9. 中药配伍协同-互补矩阵图
# ==========================================
def plot_herb_synergy_matrix(herb_names, herb_nes_matrix, file_name, result_folder="validation_results"):
    if len(herb_names) < 2 or herb_nes_matrix.shape[0] < 2: return

    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')

    corr_matrix = np.corrcoef(herb_nes_matrix)

    plt.figure(figsize=(7, 6))
    sns.set_theme(style="white", font_scale=1.1)

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="PRGn", center=0,
                xticklabels=herb_names, yticklabels=herb_names,
                cbar_kws={'label': 'Pathway Profile Correlation\n(+ : Synergy, - : Complementary)'}, 
                linewidths=1.5, linecolor='white')

    plt.title(f"Herb Synergy & Complementarity Matrix (配伍协同矩阵)\n{plot_name}", fontsize=15, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{result_folder}/{plot_name}_synergy_matrix.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# [完美修复版] 10. 靶向强度覆盖雷达图 (Radar Chart)
# ==========================================
def plot_reversal_radar_chart(vec_disease, vec_formula, file_name, result_folder="validation_results"):
    """
    绘制靶向干预强度覆盖雷达图：取绝对值，展示方剂的“火力覆盖网”是否与疾病的“破坏网”完美重合。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    mask = (vec_disease != 0)
    test = vec_disease[mask]
    sum_adjusted = vec_formula[mask]
    
    if len(test) < 6: return 
    
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')

    # 选取绝对值最大的 8 条核心通路
    top_n = min(8, len(test))
    top_idx = np.argsort(np.abs(test))[-top_n:]
    
    # 【核心修复】：全部取绝对值，展示“纯粹的作用强度”
    d_scores = np.abs(test[top_idx])
    f_scores = np.abs(sum_adjusted[top_idx])
    
    # 将强度归一化到 0 ~ 1 之间 (各自比较自己最强的那一维)
    max_d = np.max(d_scores) if np.max(d_scores) != 0 else 1
    max_f = np.max(f_scores) if np.max(f_scores) != 0 else 1
    
    d_scores = d_scores / max_d
    f_scores = f_scores / max_f
    
    labels = [f"Pathway {i+1}" for i in range(top_n)]

    angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False).tolist()
    d_scores = np.concatenate((d_scores, [d_scores[0]]))
    f_scores = np.concatenate((f_scores, [f_scores[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 【核心修复】：雷达图中心严格锁定为 0，边缘为 1.1，消除所有的畸形缩水
    ax.set_ylim(0, 1.1)
    
    # 绘制疾病破坏强度 (红色多边形)
    ax.plot(angles, d_scores, color='#EA4335', linewidth=2.5, linestyle='solid', label='Disease Perturbation Intensity')
    ax.fill(angles, d_scores, color='#EA4335', alpha=0.25)
    
    # 绘制方剂干预强度 (蓝色多边形)
    ax.plot(angles, f_scores, color='#4285F4', linewidth=2.5, linestyle='solid', label='Formula Intervention Intensity')
    ax.fill(angles, f_scores, color='#4285F4', alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    
    # 画出几圈漂亮的灰色网格线
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([]) # 隐藏数字让画面干净
    
    # 注意看这里的标题变了！跑出来的新图标题必须是 Target Engagement ... 才算替换成功！
    plt.title(f"Target Engagement & Coverage Radar\n({plot_name})", fontsize=16, pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{result_folder}/{plot_name}_radar_chart.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# ==========================================
# [新增] 11. 核心药理靶向网络图 (Bipartite Network)
# ==========================================
def plot_herb_pathway_network(herb_names, herb_nes_matrix, file_name, result_folder="validation_results"):
    """
    绘制中药-通路网络拓扑图：展示君臣佐使中药对核心通路的直接火力覆盖网络。
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import os

    if len(herb_names) < 1 or herb_nes_matrix.shape[0] < 1: return
    
    os.makedirs(result_folder, exist_ok=True)
    plot_name = file_name.replace('.csv', '').replace('.txt', '')

    # 只取前 15 条核心通路以防网络太密集变成毛线团
    top_pathways_n = min(15, herb_nes_matrix.shape[1])
    # 假设这里传入的 matrix 是针对 Top 通路的，我们截取前 15
    matrix_subset = herb_nes_matrix[:, :top_pathways_n]
    
    G = nx.Graph()
    
    # 添加中药节点 (大节点)
    for herb in herb_names:
        G.add_node(herb, type='herb', size=1500)
        
    # 添加通路节点 (小节点)
    pathway_nodes = [f"P_{i+1}" for i in range(top_pathways_n)]
    for p in pathway_nodes:
        G.add_node(p, type='pathway', size=500)

    # 添加连线 (根据作用强度)
    edges_colors = []
    edges_widths = []
    
    for i, herb in enumerate(herb_names):
        for j, p in enumerate(pathway_nodes):
            weight = matrix_subset[i, j]
            if abs(weight) > 0.1: # 过滤掉微弱连接
                G.add_edge(herb, p, weight=weight)
                edges_colors.append('#EA4335' if weight > 0 else '#4285F4') # 红激活，蓝抑制
                edges_widths.append(abs(weight) * 5) # 线条粗细代表力度

    plt.figure(figsize=(10, 10))
    
    # 使用二分图布局 (Bipartite Layout) 或 弹簧布局 (Spring Layout)
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    # 分离节点类型画图以设置不同颜色
    herb_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'herb']
    pathway_nodes_in_g = [n for n, d in G.nodes(data=True) if d['type'] == 'pathway']
    
    nx.draw_networkx_nodes(G, pos, nodelist=herb_nodes, node_color='#FBBC05', node_size=1500, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=pathway_nodes_in_g, node_color='lightgray', node_size=600, edgecolors='gray')
    
    nx.draw_networkx_edges(G, pos, width=edges_widths, edge_color=edges_colors, alpha=0.7)
    
    # 节点标签
    nx.draw_networkx_labels(G, pos, font_family=plt.rcParams['font.sans-serif'][0], font_size=10, font_weight='bold')

    plt.title(f"Herb-Pathway Pharmacological Network\n({plot_name})", fontsize=16, pad=20)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{result_folder}/{plot_name}_network_graph.pdf", format="pdf", bbox_inches="tight")
    plt.close()