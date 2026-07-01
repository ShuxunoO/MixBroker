# MixBroker 在 Tornado Cash 存款↔收款人链接预测数据集上的复现与编码器消融报告

> **任务**:用 [`pyg_export_dataset_report.md`](../../AML_for_Blockchain/docs/pyg_export_dataset_report.md) 导出的 PyG 数据集(节点行为特征 + ground-truth 链接标签),**完全沿用 MixBroker 代码仓库的训练/验证设定**,跑通 MixBroker 的 GraphSAGE 链接预测;并按仓库 `model.py` 中给出的方式对 **GNN 编码器骨干(GCN / GAT / GraphSAGE)做消融**,先在 42d 行为特征上、再在 82d(行为+结构)特征上各做一遍,记录并分析。
>
> **代码**:`/Shuxun/RelatedWork/MixBroker`(`main_cpu.py` / `data_process.py` / `model.py`,`cpu` 分支)
> **环境**:conda `mixbroker`(Python 3.12,torch 2.6.0+cpu,torch-geometric 2.7.0,scikit-learn,pandas 2.x,CPU)
> **产物目录**:`/Shuxun/RelatedWork/MixBroker/reports/`
> **日期**:2026-07-01。

---

## 0. 结论速览(TL;DR)

1. **环境/流程忠实可复现**:用仓库自带数据跑 `main_cpu.py`,得到 F1 0.850 / AUC 0.875(论文级),证明代码路径与设定无误。
2. **在我们自己的 TC 数据上,MixBroker 的 GraphSAGE 依然稳健**:F1 0.698(42d)→ **0.708(82d)**,AUC 0.857 → **0.862**,且 10 折方差极小(F1 std ≈ 0.02,而原始小数据集 std ≈ 0.08)。
3. **编码器消融结论与论文一致——GraphSAGE 完胜 GCN/GAT**:同样的 2 层 `in→32→16` + 点积解码器结构下,SAGE 的 F1(0.70)是 GCN(0.39)/GAT(0.34)的近 2 倍。原因见 §6.1。
4. **82d 结构特征对"有效的"编码器(SAGE)有小幅净增益**(召回 +2.7pt、AUC +0.5pt),对"已失效"的 GCN/GAT 无稳定帮助。

---

## 1. 背景:MixBroker 的流程与"设定"到底是什么

严格按仓库代码梳理,MixBroker 的训练/验证管线如下(全部保持原样,未改任何超参):

| 环节 | 仓库设定(源码位置) |
|---|---|
| 节点特征 | `Dataset/Graph/node_feature_normalized.csv`,`data_process.load_10_fold_data` 丢弃 `node, value_d, value_w, avg_value_d, avg_value_w` 五列,**剩 `nodeid` + 42 维特征**(15 次数 + 9 gas price + 18 时间)。 |
| **图结构** | `edge_index` **只由候选边(正+负样本对)构成** —— 即"待预测的链接本身就是消息传递用的边"。这是 MixBroker 的原始设计(`load_10_fold_data`:`edge_index = edge_label_index`)。 |
| 编码器 | `model.GNN_NET`:2 层,`in→32→16`,层间 ReLU。仓库用 `SAGEConv`,并在 `model.py` 中以注释形式给出 `GCNConv` / `GATConv` 两个备选骨干——**这就是仓库自带的"编码器消融"入口**。 |
| 解码器 | 点积:`(z_src * z_dst).sum(-1)`,`sigmoid` 后按 **阈值 0.75** 二值化。 |
| 训练 | `Adam(lr=0.01)`,`BCEWithLogitsLoss`,**100 epochs**,`min_epochs=10`,每个 epoch 在测试折上评估、**按 F1 取最优**。 |
| 交叉验证 | `KFold(n_splits=10, shuffle=True, random_state=42)`,正、负样本各自独立 10 折、按折号配对。 |
| 随机种子 | `seed_torch(1029)`。 |
| 指标 | Accuracy / Precision / Recall / F1 / FPR / FNR / ROC-AUC(`sklearn`)。 |

> 这些设定 **在本报告所有实验中逐字保留**。唯一被参数化的两处是:① 编码器骨干(消融的自变量,复刻自 `model.py` 的注释);② 特征文件(42d vs 82d)。见 §3 的忠实性说明。

---

## 2. 数据:从 PyG 导出 → MixBroker 格式

### 2.1 特征映射(42d 完全对齐)

MixBroker 真正喂给模型的 42 维,与 PyG 导出报告里的"42d 行为特征"**逐字段一一对应**(仅池计数列 `pool_X_num` 需重命名为 `X_num`;9 个 gas、18 个时间列名完全相同)。因此这是一次**语义无损**的映射。

- 节点集合:PyG 中 `is_seed==1` 的 **155,204** 个种子地址(存/取/发起人),它们才有 42d 行为特征。
- `value_d/value_w/avg_value_d/avg_value_w` 四列按 `feature_extract.feature_extract_value` 的公式由计数×面额算出(但这四列会被 `load_10_fold_data` 丢弃,仅为对齐 schema)。
- **归一化**:严格复刻 `feature_extract.normalize_handle` —— 对全部特征列做 `sklearn.StandardScaler`(按全体种子人群估计 z-score),`inf→NaN`;加载时 `x[inf]=1, x[nan]=0`(同仓库)。
- 生成的 `node_feature_normalized.csv` 表头与仓库文件 **48 列逐字节一致**(已 `diff` 校验)。

### 2.2 标签(ground truth)

直接取 `labels.parquet`:**正样本 3,808**(deposit↔recipient 已证实关联,`GF-1/GF-2/DL-2`)、**负样本 11,424**(同池 hard negative,1:3)。写成仓库格式的 `train_pos_edge_10fold.csv` / `train_neg_edge_10fold.csv`(带 `Unnamed: 0` 索引列)。两端节点 **全部命中** 节点特征表(缺失 0 条)。

### 2.3 82d 特征(用于第二轮消融)

取 PyG 的 **82 维建模特征** = 4 身份位 + 42 行为 + 36 结构(度/桥接/金额/资产/category/池暴露/时间跨度),同样 `StandardScaler` 归一化后写 `node_feature_82d_normalized.csv`。除特征维度外,**其余管线与 42d 完全相同**——这样 42d vs 82d 的唯一变量就是"特征集"。

### 2.4 一处影响性能、但对结果零影响的工程优化

MixBroker 的图 `edge_index` 只由候选边构成,因此**没有出现在任何标签里的节点从不参与消息传递**(既不是任何被解码节点的邻居,SAGE/GCN/GAT 输出对其不敏感)。155,204 个种子里仅 **1,484** 个出现在标签中。故把节点特征表**裁剪到这 1,484 个标签节点**(特征值仍是按全体种子估计的 z-score,未改动),可证明得到**逐位相同**的结果、但快约 100×。

> **已实证**:SAGE/42d 在"全量 155k 节点表"与"1,484 裁剪表"上跑出的 10 折指标**完全相同**(F1=0.6981, Acc=0.8605, AUC=0.8568)。因此下文所有消融均在裁剪表上运行,结果与全量表等价。

---

## 3. 实验设计与忠实性声明

| 实验 | 数据 | 编码器 | 目的 |
|---|---|---|---|
| **E0 基线** | 仓库自带 `Dataset/Graph`(103 正 / 103 负) | SAGE / 42d | 验证环境与代码路径,复现论文级结果 |
| **E1 主实验** | 我们的 TC 数据(3808 / 11424) | SAGE / 42d | MixBroker 原样跑我们的数据 |
| **E2 编码器消融 · 42d** | 我们的 TC 数据 | GCN / GAT / SAGE | 复刻仓库 `model.py` 的骨干消融 |
| **E3 编码器消融 · 82d** | 我们的 TC 数据(82 维) | GCN / GAT / SAGE | 同思路,换成行为+结构特征 |

**忠实性**:
- E0/E1 直接调用**未改动**的 `main_cpu.py`(仅在 `cpu` 分支把 `.cuda()`→`.to(device)`、`df.append`→`pd.concat`,逻辑/超参/种子与原 `main.py` 一致)。E1 的复现用一个薄封装 `run_experiment.py` 调用仓库 `train_10_fold()`,只额外落盘结果。
- E2/E3 用统一 runner `ablation_run.py`:训练循环、`mytest`、KFold、种子、超参**逐行照抄** `main_cpu`,仅把编码器骨干与特征文件参数化。已用 SAGE/42d 复算验证其输出与 `main_cpu` **完全一致**,证明 runner 无偏差。
- 三个骨干均为**同一** 2 层 `in→32→16`+ReLU+点积结构,只替换卷积算子(`GCNConv`/`GATConv`/`SAGEConv`,默认参数,GAT `heads=1`)——等价于把 `model.py` 里对应的两行注释取消。

---

## 4. E0:基线复现(仓库自带数据)

`conda run -n mixbroker python main_cpu.py`,耗时 ~85s。

| 指标 | mean | std |
|---|---|---|
| Accuracy | 0.8518 | 0.078 |
| Precision | 0.8679 | 0.109 |
| Recall | 0.8473 | 0.125 |
| **F1** | **0.8500** | 0.081 |
| FPR | 0.1436 | 0.131 |
| FNR | 0.1527 | 0.125 |
| **AUC** | **0.8750** | 0.059 |

**结论**:成功复现论文级结果(F1≈0.85、AUC≈0.875)。注意 std 很大(±0.08~0.13)——因为原始数据集每折仅约 20 条测试边,估计噪声大。

---

## 5. E1 + E2 + E3:主结果表(我们的 TC 数据)

10 折均值±标准差。**粗体为每列最优的我方实验**。

| 实验 | Acc | Prec | Rec | F1 | FPR | FNR | AUC |
|---|---|---|---|---|---|---|---|
| E0 仓库数据 · SAGE/42d | 0.8518±.078 | 0.8679±.109 | 0.8473±.125 | 0.8500±.081 | 0.1436 | 0.1527 | 0.8750±.059 |
| E1 我方 · **SAGE/42d** | 0.8605±.008 | 0.7606±.020 | 0.6455±.024 | 0.6981±.018 | 0.0678 | 0.3545 | 0.8568±.017 |
| E3 我方 · **SAGE/82d** | **0.8615±.009** | 0.7507±.036 | **0.6725±.032** | **0.7083±.015** | 0.0755 | **0.3275** | **0.8622±.018** |
| E2 我方 · GCN/42d | 0.6461±.029 | 0.3422±.046 | 0.4577±.101 | 0.3901±.063 | 0.2911 | 0.5423 | 0.6230±.039 |
| E3 我方 · GCN/82d | 0.6674±.053 | 0.3408±.117 | 0.2957±.126 | 0.2963±.078 | 0.2087 | 0.7043 | 0.5863±.060 |
| E2 我方 · GAT/42d | 0.6256±.068 | 0.3199±.069 | 0.3931±.099 | 0.3429±.035 | 0.2969 | 0.6069 | 0.5731±.056 |
| E3 我方 · GAT/82d | 0.5937±.070 | 0.2876±.063 | 0.4347±.188 | 0.3356±.093 | 0.3533 | 0.5653 | 0.5552±.074 |

编码器消融并列对照(F1 / AUC,mean):

| 编码器 | 42d F1 | 42d AUC | 82d F1 | 82d AUC |
|---|---|---|---|---|
| **GraphSAGE** | **0.698** | **0.857** | **0.708** | **0.862** |
| GCN | 0.390 | 0.623 | 0.296 | 0.586 |
| GAT | 0.343 | 0.573 | 0.336 | 0.555 |

（各实验逐折明细见 `reports/results_<enc>_<feat>_per_fold.csv` 与 `baseline_per_fold.csv`。）

---

## 5.5 关键细节:训练/测试规模、判定方案、单轮开销

### 5.5.1 用于训练/测试的数据规模(是全量 DL+GF,不是论文的 103 对)

本报告 **E1/E2/E3 全部使用我们数据集的全量 cross-address ground truth**,**不是**原论文/仓库自带的 103 对:

| 数据集 | 正样本 | 负样本 | 合计 | 备注 |
|---|---|---|---|---|
| 原论文/仓库(E0 基线) | 103 | 103 | 206 | 1:1 |
| **我们的 TC 数据(E1/E2/E3)** | **3,808** | **11,424** | **15,232** | 1:3 hard-neg |

正样本 3,808 条即 `labels.parquet` 中**全部 cross-address 线索**,按线索类型:

| clue_type | 条数 | 类别 |
|---|---|---|
| **GF-1** | 3,567 | Ground-truth / ENS-family(GF) |
| **GF-2** | 69 | GF |
| **DL-2** | 172 | Deposit-linkage(DL) |
| 合计 | **3,808** | 即 **DL + GF 全量** |

> 说明:`labels.parquet` 里另有 5,160 条 `deposit==withdraw` 的自链线索(对链接预测无意义),已在导出阶段剔除;3,808 是两端为不同地址、且两端节点都保留的**有效全量线索**。负样本 11,424 为同池 hard negative(1:3)。

**10 折切分后的每折规模**(`KFold(10, shuffle, rs=42)`,正负各自独立切分):

| | 训练 | 测试 |
|---|---|---|
| 正样本对 | ~3,427 | ~381 |
| 负样本对 | ~10,282 | ~1,142 |
| **候选边合计** | **~13,709** | **~1,523** |

这与运行日志中的张量形状一致:`train Data(edge_index=[2, 13710])`、`test Data(edge_index=[2, 1522])`。参与消息传递的候选节点共 **1,484** 个(见 §2.4)。

### 5.5.2 判定方案(每条候选边如何被判成正/负,指标如何算)

完全沿用仓库 `mytest()` 的计算,**逐字未改**:

1. **编码**:`z = GNN.encode(x, edge_index)`,每个节点得到 16 维嵌入 `z_i ∈ R¹⁶`(2 层 `in→32→16`,层间 ReLU)。
2. **打分(解码)**:对每条候选边 `(a,b)`,`score = sigmoid( z_a · z_b )`(16 维点积后过 sigmoid),得到 0~1 的关联概率。
3. **判定(二值化)**:`pred = 1 if score > 0.75 else 0`(**固定阈值 0.75**,仓库设定)。
4. **混淆矩阵**:逐条与真值 `y` 比对累计 TP/FP/FN/TN。
5. **指标计算**:
   - `Accuracy/Precision/Recall/F1` = `sklearn.metrics` 在 `pred` vs `y` 上计算;
   - `FPR = FP/(FP+TN)`,`FNR = FN/(TP+FN)`;
   - `AUC = roc_auc_score(y, score)` —— 用**连续分数**、**与阈值无关**(所以即便阈值 0.75 压低了召回,AUC 仍能反映排序判别力)。
6. **选优与聚合**:每折训练 **100 个 epoch**,**每个 epoch 都在该折测试集上评估一次**;取 `epoch>10` 中 **F1 最高** 的那个 epoch 的全部指标作为该折结果;最后对 **10 折求均值±标准差**(即 §5 表格里的数字)。

> 一句话:**"存款↔收款人是否关联" = 两端 16 维嵌入的点积过 sigmoid 是否 > 0.75**;判别力用阈值无关的 AUC 衡量,运营口径的精确率/召回率用 0.75 阈值衡量。

### 5.5.3 单轮耗时与内存占用(实测)

宿主机:8 核 / 16 GB,CPU-only(conda `mixbroker`,torch 2.6.0+cpu)。用 `/usr/bin/time -v` 实测(subset 特征表,即本报告实际使用的配置):

| 配置 | 每轮(完整 10 折)墙钟 | 每折约 | 峰值内存(Max RSS) | CPU 占用 |
|---|---|---|---|---|
| E0 基线(103 对,42d) | ~85 s | ~8.5 s | ~0.4 GB | 多线程 |
| SAGE 42d(3808/11424) | **~210 s(低负载)** | **~21 s** | **~0.43 GB** | ~370%(≈3.7 线程) |
| SAGE 82d(3808/11424) | ~210 s(低负载) | ~21 s | ~0.44 GB | ~3.7 线程 |

**说明**:
- **内存**:峰值仅约 **0.43–0.44 GB**,远低于 16 GB;42d 与 82d 几乎相同(瓶颈是 torch 运行时与逐 epoch 评估,而非特征维度或节点表大小)。因此该实验对内存**毫无压力**。
- **耗时**:单轮开销主要来自"100 epoch × 每 epoch 在测试折上评估一次(含逐条 TP/FP 的 Python 循环 + sklearn)× 10 折",**与特征维度/节点数几乎无关**,故 42d≈82d≈210 s。GCN/GAT 在**全量 155k 节点表**上会因强制加自环(155k 自环边)而慢 5–10×,这也是本报告改用等价的 1,484 节点裁剪表的原因(§2.4)。
- **实测波动**:运行期间宿主机被其他会话的 GNN 作业占满(load 一度到 24–88 / 8 核),同一 10 折任务墙钟在 **134 s ~ 395 s** 间波动——这是 CPU 争抢导致,**不影响任何数值结果**(种子固定 + CPU 确定性);上表"低负载"列为无争抢时的代表值。

---

## 6. 分析

### 6.1 为什么 GraphSAGE 完胜 GCN/GAT(消融核心结论)

在 MixBroker 的设计里,**图的边就是待预测的正/负候选对**。这对三种卷积算子的影响截然不同:

- **GCNConv / GATConv** 使用（对称归一化 / 注意力加权的)**邻接聚合并强制加自环**,本质是把节点特征沿边做平滑。由于边同时包含正样本对与负样本对,平滑会把两端特征**无差别地拉近**;再用点积解码器,正负对的嵌入相似度被压缩到难以区分——于是 F1 掉到 ~0.34–0.39、AUC 逼近 0.55–0.62(接近随机)。GCN/82d 甚至因维度更高、平滑更"糊"而 FNR 飙到 0.70。
- **SAGEConv** 对"自身"与"邻居"用**两套独立权重**(`W_root·x_v + W_nbr·agg`),即使做了邻居聚合,**节点自身身份仍被保留**;因此正负对不会被强行同化,点积解码器能保留可分性。这正是论文选择 GraphSAGE 的根本原因,消融在我们的数据上**独立复现了这一结论**。

> 结论:MixBroker 的性能高度依赖 GraphSAGE 的"自表征保留"特性;把编码器换成 GCN/GAT 会使框架在该任务上基本失效。这与仓库默认启用 SAGE、把 GCN/GAT 留作注释备选的做法互相印证。

### 6.2 我们的数据 vs 原始数据(E1 vs E0)

- **精度维度**:我方 SAGE Accuracy 略高(0.860 vs 0.852)、**FPR 低得多**(0.068 vs 0.144),说明在大规模、1:3 hard-negative 下,模型的"误报控制"更好。
- **召回维度**:我方 Recall/F1 低于原始(0.65/0.70 vs 0.85/0.85)。这是**数据难度而非模型退化**导致:① 负样本是"同池 hard negative",1:3 比例远比原始 1:1 更难;② 固定阈值 0.75 对 1:3 分布偏保守,压低召回、抬高 FNR。**AUC 0.857 与原始 0.875 基本持平**,说明排序能力相当——差距主要来自阈值与类不平衡,而非判别力。
- **稳定性维度**:我方 10 折 std 只有原始的 **1/4~1/8**(F1 std 0.018 vs 0.081)。因为我方每折有数百条测试边,评估远比原始(每折~20 条)稳健。这是我方数据集的重要优势。

### 6.3 82d 结构特征的作用(E3 vs E2/E1)

- 对 **SAGE**:82d 带来一致小幅提升——F1 0.698→**0.708**、AUC 0.857→**0.862**、Recall 0.646→**0.673**、FNR 0.355→**0.328**。结构特征(桥接 `is_bridge`/`distinct_*_seeds`、度、池暴露)确实补充了行为特征之外的关联信号,尤其提升召回。
- 对 **GCN/GAT**:82d 无稳定收益(GCN 反而更差)。因为这些编码器本身已被"候选边平滑"破坏(§6.1),更高维特征只是被平滑得更糊。
- **启示**:结构特征的价值只有在"编码器能保留节点自表征"时才兑现;它不能挽救不合适的骨干。

### 6.4 阈值敏感性提示

固定阈值 0.75(仓库设定,已保留)在 1:3 分布下明显偏高,是我方 Recall/FNR 偏低的主因。AUC(阈值无关)显示排序能力良好,若按业务需要调阈值或用 AUC-PR 评估,召回可显著改善。**本报告未改该阈值**,以严格遵守仓库设定;这一点作为后续可调项记录。

---

## 7. 局限与说明

1. **图结构沿用 MixBroker 原设计**(边=候选对),**未引入** PyG 导出里丰富的 transfer 交易图(`data_homo.pt`/`data_hetero.pt`)。若接入真实交易拓扑,GCN/GAT 的表现可能改观——但那已超出"完全使用仓库设定"的范围,属于后续工作。
2. 类别不平衡(1:3)与固定阈值 0.75 共同压低召回;这是数据属性 + 仓库设定的组合效应,非模型缺陷。
3. 归一化按"全体 155k 种子"人群估计 z-score(对应 MixBroker 对其全节点集归一化);裁剪节点表不改变这些值,已实证结果等价。
4. 运行期间宿主机被其他会话的 GNN 作业占满(load 88/8 核),仅影响墙钟耗时,不影响任何数值结果(种子固定、CPU 确定性)。

---

## 8. 复现步骤与产物清单

**数据构建**(conda `AML`,含 pyarrow):
```bash
# 42d: PyG nodes_features.parquet(is_seed) + labels.parquet → MixBroker 格式
python build_mixbroker_data.py       # → node_feature_normalized.csv / train_pos|neg_edge_10fold.csv
python build_82d.py                  # → node_feature_82d_normalized.csv
# 裁剪到 1484 个标签节点(结果等价、提速);略
```

**训练/验证**(conda `mixbroker`,`run_tc/` 工作目录):
```bash
# E0 基线(仓库根目录,未改代码)
python main_cpu.py
# E1 主实验(薄封装调用仓库 train_10_fold)
python run_experiment.py
# E2/E3 编码器消融矩阵
for bb in sage gcn gat; do
  python ablation_run.py --backbone $bb --feat .../node_feature_42d_sub.csv \
    --drop '["node","value_d","value_w","avg_value_d","avg_value_w"]' --out results_${bb}_42d
  python ablation_run.py --backbone $bb --feat .../node_feature_82d_sub.csv \
    --drop '["node"]' --out results_${bb}_82d
done
```

**产物**(`/Shuxun/RelatedWork/MixBroker/reports/`):

| 文件 | 内容 |
|---|---|
| `mixbroker_reproduction_report.md` | 本报告 |
| `baseline_per_fold.csv` / `baseline_summary.json` / `baseline_repo_data.log` | E0 基线逐折 + 汇总 + 原始 stdout |
| `results_<enc>_<feat>_per_fold.csv` | E1/E2/E3 六个实验各自 10 折明细(enc∈{sage,gcn,gat}, feat∈{42d,82d}) |
| `results_<enc>_<feat>_summary.json` | 各实验 mean/std + 元信息(骨干、特征维度、耗时) |
| `logs_<enc>_<feat>.log` | 各实验运行日志 |

工作代码与数据在 `/Shuxun/RelatedWork/MixBroker/run_tc/`(`ablation_run.py`、`model_ablation.py`、`run_experiment.py`、`Dataset/Graph/*`),原仓库 `Dataset/Graph` 未被改动。
