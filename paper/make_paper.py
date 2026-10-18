"""生成技术报告 docx。

    python paper/make_paper.py
"""
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from docx.oxml import OxmlElement

DOC = Document()


def _set_east_asia(rpr, name):
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)


def set_ea(run, name):
    _set_east_asia(run._element.get_or_add_rPr(), name)


# ---------- 全局字体 ----------
style = DOC.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(10.5)
_set_east_asia(style.element.get_or_add_rPr(), "宋体")
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.line_spacing = 1.25


def h(text, level=1):
    p = DOC.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Times New Roman"
        set_ea(run, "黑体")
    return p


def para(text, size=10.5, bold=False, align=None, italic=False):
    p = DOC.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if align:
        p.alignment = align
    return p


def caption(text):
    p = DOC.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(9)
    run.italic = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def table(header, rows, cap):
    t = DOC.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, cell_text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(cell_text)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(9)
    caption(cap)


def figure(path, width_cm, cap):
    DOC.add_picture(path, width=Cm(width_cm))
    DOC.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(cap)


# ---------- 标题 ----------
title = DOC.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Sparse-RADDT：L1 稀疏正则化可微分决策树的复现与受控分析")
run.bold = True
run.font.size = Pt(16)
set_ea(run, "黑体")

para("睢昊霖　北京邮电大学 国际学院", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
para("技术报告 · 2026 年 10 月", size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

# ---------- 摘要 ----------
h("摘要", 1)
para(
    "决策树以可解释性见长，但贪心构建策略只能保证局部最优。RADDT 通过"
    "“ReLU + Argmin”重构，把整棵树的训练转化为无约束可微优化，在保留硬分裂的同时实现全局优化。"
    "本文做两件事：其一，独立复现 RADDT，并在四个 UCI 数据集上与 CART、随机森林做五折交叉验证对比；"
    "其二，在分裂权重上引入 L1 稀疏正则（Sparse-RADDT），并系统考察树深与正则强度的影响。"
    "实验表明：修正树深设置后，RADDT 在四个数据集上均优于 CART，并在 Wine 与 BreastCancer 上追平随机森林；"
    "L1 正则的效果强烈依赖强度——λ=1e-3 时在 Digits 上同时提升准确率（+1.2 个百分点）与权重稀疏度（7.0%），"
    "而 λ=1e-2 会造成 6.2 个百分点的性能下降。本文进一步记录了复现过程中四个容易被忽视的实现陷阱，"
    "它们都会让实验结果在表面正常的情况下失去意义。"
)
para("关键词：可微分决策树；RADDT；L1 稀疏正则；复现研究；机器学习工程", size=10)

h("Abstract", 1)
para(
    "Decision trees are valued for interpretability, but greedy induction is only locally optimal. "
    "RADDT reformulates tree training as an unconstrained differentiable optimization problem via a "
    "ReLU + Argmin construction, keeping hard splits while enabling global optimization. "
    "We (i) independently reproduce RADDT and compare it with CART and Random Forest on four UCI datasets "
    "under 5-fold cross-validation, and (ii) add an L1 penalty on split weights (Sparse-RADDT) and study "
    "the effect of tree depth and regularization strength. After correcting the tree-depth configuration, "
    "RADDT outperforms CART on all four datasets and matches Random Forest on Wine and BreastCancer. "
    "The L1 penalty is highly sensitive to its strength: at lambda=1e-3 it improves Digits accuracy by "
    "1.2 points while sparsifying 7.0% of the weights, whereas lambda=1e-2 costs 6.2 points. "
    "We also document three implementation pitfalls that silently invalidate results while appearing to work."
)

# ---------- 1 引言 ----------
h("1  引言", 1)
para(
    "在医疗诊断、信贷审批等高风险场景中，模型的可解释性与准确率同等重要。决策树把决策过程表达为"
    "一系列 IF-THEN 规则，是最具代表性的可解释模型[2]。但 CART 一类的贪心构建策略在每个节点只做局部最优分裂，"
    "整体上无法保证全局最优；而随机森林[3]、XGBoost[5] 这类集成方法虽然精度更高，却牺牲了单棵树的可读性。"
)
para(
    "RADDT[1] 提出了一条折中路线：把整棵树重构成一个连续优化问题，使得分裂参数、阈值与叶节点预测都能通过"
    "梯度下降联合优化，同时推理时仍然执行硬分裂。本文的工作包括三点："
)
para("（1）独立复现 RADDT，并在统一的评测协议下与 CART、随机森林对比；")
para("（2）提出 Sparse-RADDT：在分裂权重上施加 L1 惩罚，使树在训练中自动进行特征选择，并系统分析正则强度的影响；")
para("（3）记录复现过程中四个实现陷阱——它们不会报错，却会让实验结论失效。我们认为这一节对后续复现者最有价值。")

# ---------- 2 背景 ----------
h("2  背景：RADDT", 1)
para(
    "设树深为 D，则叶节点数 T = 2^D，内部节点数 T − 1。每个内部节点 j 拥有分裂权重 a_j 与阈值 b_j。"
    "样本 x 沿路径走向某个叶节点 t 时，每经过一个祖先节点都会产生一个“违反量”：若该节点要求左转（a_j^T x ≤ b_j）"
    "而实际偏右，违反量为 ReLU(a_j^T x − b_j)；反之则为 ReLU(b_j − a_j^T x)。把路径上所有祖先的违反量累加，"
    "得到 U(x, t)。RADDT 的关键性质是：只有正确叶节点的累积违反量为零，因此 argmin_t U(x, t) 即为路由结果。"
)
para(
    "由于 argmin 不可微，训练时用 scaling softmin 近似：p(t | x) ∝ exp(−α U(x, t))，其中 α 从 2 退火到 200。"
    "α 较小时梯度平滑、易于优化；α 增大后分布逐渐逼近 one-hot，最终与推理时的硬路由一致。"
    "训练目标为按 p(t | x) 加权的叶节点预测损失。"
)

# ---------- 3 方法 ----------
h("3  Sparse-RADDT", 1)
para(
    "在 RADDT 中，每个内部节点都用全部特征参与分裂，分裂权重矩阵 A 通常是稠密的。这与决策树“只用少数特征就能决策”"
    "的设计哲学相悖，也削弱了可解释性。受 Lasso[4] 启发，我们在损失中加入 L1 惩罚："
)
para("L = L_task + λ · ‖A‖₁", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para(
    "梯度下降会把不重要的分裂权重压向零，从而在训练中完成特征选择。需要强调的是：L1 与 Adam 的组合并不保证"
    "产生精确的零权重，因此本文报告的稀疏度是权重绝对值小于 1e-6 的比例，而不是可视化时的阈值划分。"
)

# ---------- 4 实验 ----------
h("4  实验设置", 1)
para(
    "数据集：Iris（150×4，3 类）、Wine（178×13，3 类）、BreastCancer（569×30，2 类）、Digits（1797×64，10 类），"
    "均为 scikit-learn 内置数据。评测协议：分层五折交叉验证，随机种子 42，报告准确率均值与标准差。"
)
para(
    "模型配置：树深设为 depth = max(3, ⌈log₂ K⌉)，其中 K 为类别数，以保证叶节点数不少于类别数；"
    "学习率 0.01，每个退火阶段 150 轮，α 从 2 到 200 共 4 个阶段，2 次随机重启，每次重启固定随机种子。"
    "基线为 CART（默认参数）与随机森林（100 棵树）。Sparse-RADDT 与 RADDT 除 λ 外配置完全相同。"
)

# ---------- 5 结果 ----------
h("5  结果", 1)
h("5.1  主结果", 2)
figure("figures/accuracy.png", 15.5, "图 1　四个数据集上的五折交叉验证准确率（λ=1e-3）")
table(
    ["数据集", "类别", "树深", "CART", "随机森林", "RADDT", "Sparse-RADDT", "稀疏度"],
    [
        ["Iris", 3, 3, "0.953", "0.947", "0.960", "0.967", "21.4%"],
        ["Wine", 3, 3, "0.893", "0.978", "0.978", "0.977", "15.4%"],
        ["BreastCancer", 2, 3, "0.910", "0.956", "0.974", "0.965", "11.9%"],
        ["Digits", 10, 4, "0.855", "0.978", "0.882", "0.894", "7.0%"],
    ],
    "表 1　主实验结果（准确率）",
)
para(
    "RADDT 在四个数据集上全部优于 CART，在 Wine 与 BreastCancer 上追平随机森林（0.978 对 0.978，0.974 对 0.956）。"
    "Digits 上仍明显落后于随机森林，这符合预期：单棵树的预测方差无法通过平均来消除，属于结构差异而非训练不足。"
)
para(
    "Sparse-RADDT 在 Iris 与 Digits 上带来提升（+0.7、+1.2 个百分点），在 Wine 上基本持平，在 BreastCancer 上下降 0.9 个百分点。"
    "这说明 L1 的收益依赖于数据的高维程度与特征冗余结构，并非普适。"
)

h("5.2  树深必须满足叶子数不少于类别数", 2)
figure("figures/depth_study.png", 11, "图 2　Digits 上不同树深的准确率（固定 λ=0、固定种子）")
table(
    ["树深", "叶子数", "准确率"],
    [["2", 4, "0.380"], ["3", 8, "0.681"], ["4", 16, "0.882"]],
    "表 2　树深对 Digits 的影响（10 类）",
)
para(
    "这是一个受控对照：只改树深，其余因素全部固定。8 个叶子无法覆盖 10 个类别，准确率被结构性压在 0.68；"
    "扩到 16 个叶子后升至 0.88。该约束可以由类别数直接算出，却很容易在复现中被忽略。"
)

h("5.3  正则强度：不是越强越好", 2)
figure("figures/lambda_sweep.png", 11, "图 3　L1 强度对 Digits 准确率与权重稀疏度的影响")
table(
    ["λ", "准确率", "稀疏度"],
    [["0", "0.882", "0.0%"], ["1e-4", "0.878", "4.1%"], ["1e-3", "0.894", "7.0%"], ["1e-2", "0.819", "9.8%"]],
    "表 3　正则强度扫描（Digits）",
)
para(
    "λ=1e-3 是一个甜点：稀疏度达到 7.0% 的同时，准确率反而比无正则高 1.2 个百分点。"
    "而 λ=1e-2 会过度稀疏化，准确率下降 6.2 个百分点。可见“加正则”本身并不产生收益，"
    "强度的选择才是关键。"
)
para(
    "需要说明的是，λ=1e-3 是在这同一组交叉验证折上选出来的，因此 +1.2 个百分点应视为探索性结果，""而非无偏估计；严格做法是嵌套交叉验证或独立的验证集，本文没有做。"
)

h("5.4  自适应正则强度：直觉与实现相反", 2)
table(
    ["数据集", "特征维度", "固定 λ=1e-3", "自适应 λ₀=1e-3"],
    [["Iris", 4, "0.967", "0.980"], ["Wine", 13, "0.977", "0.978"],
     ["BreastCancer", 30, "0.965", "0.961"], ["Digits", 64, "0.894", "0.858"]],
    "表 4　固定强度与自适应强度的对比",
)
para(
    "原方案提出按特征维度调整强度：λ = λ₀·ln(1+p)/√p，直觉是“高维需要更强正则”。但把它真正实现之后，"
    "在这个维度区间里得到的是**递减**的等效强度：p=4 时约 8.0e-4，p=64 时约 5.2e-4。"
    "实验也印证了这一点——自适应在 4 维的 Iris 上提升 1.3 个百分点，在 64 维的 Digits 上反而下降 3.6 个百分点。"
    "因此我们认为该方案的有效性依赖维度区间，不宜作为默认配置。"
)

h("5.5  可解释性与收敛性", 2)
figure("figures/feature_importance.png", 14.5, "图 4　Wine 上 RADDT 与 Sparse-RADDT 的特征重要性（前 10 维）")
para(
    "加入 L1 之后，权重明显向少数特征集中：f0、f11、f6 的重要性上升，而 f4、f5 被压到接近零。"
    "这与表 1 中 Wine 上 15.4% 的权重稀疏度一致——稀疏化确实在做特征选择，而不是均匀地缩小所有权重。"
)
figure("figures/decision_boundary.png", 14.5, "图 5　Iris 前两个特征上的决策边界对比")
para(
    "在二维子空间里，两种模型的边界形状接近，说明在这个维度上稀疏化没有牺牲表达能力。"
    "这也解释了为什么 Iris 上 Sparse-RADDT 的准确率不降反升。"
)
figure("figures/convergence.png", 12, "图 6　λ 对训练收敛的影响（Digits，两次重启拼接）")
para(
    "λ=0 在前期收敛更快，但 λ=1e-3 最终收敛到更低的损失，与 5.3 节“稀疏版准确率反而更高”的结论一致。"
    "曲线中的两次跳变来自两次随机重启——每次重启都会重置训练损失，这也是原文采用多次重启选最优的原因。"
)

# ---------- 6 陷阱 ----------
h("6  复现中的四个陷阱", 1)
para(
    "我们的第一版实现可以跑通、也能产出“看起来合理”的表格，但结论是错的。以下四点是根因，且都不会触发任何报错。"
)
para(
    "陷阱一：scikit-learn 的 clone 会静默丢弃参数。当估计器的 __init__ 用 **kwargs 接收超参数时，"
    "get_params() 只能看到显式签名中的参数，交叉验证在克隆估计器时会把 n_alpha_stages、n_restarts 等"
    "恢复成默认值。结果是：脚本里写着 4 个退火阶段、2 次重启，实际跑的是 5 个阶段、3 次重启。"
    "修正方式是全部改为显式参数，并加入“克隆后参数是否还在”的回归测试。"
)
para(
    "陷阱二：叶子数少于类别数。第一版对所有数据集统一使用 depth=3，即 8 个叶子，而 Digits 有 10 个类别。"
    "模型在结构上就无法表达 10 个类别，指标被压缩，却被误读为“该方法在高维任务上表现不佳”。"
)
para(
    "陷阱三：对照实验中混入第二个变量。修正树深的同时，正则强度也发生了改变，导致“提升来自树深”这一说法不成立。"
    "本文第 5.2 节的实验固定了 λ 与随机种子，只改变树深，才构成受控对照。"
)
para(
    "陷阱四：估计器没有被 scikit-learn 认成分类器。分类器的继承顺序写成 (RADDTBase, ClassifierMixin)，mixin 排在后面，"
    "is_classifier() 因此返回 False。cross_val_score 收到整数 cv 时会退回普通 KFold；Iris 按类别排序，每一折整折都是同一类，"
    "训练集里根本没有这个类别，准确率直接变成 0。显式传 StratifiedKFold 时一切正常，所以这个 bug 只有在换一种调用方式时才会暴露。"
    "修正方式是把 mixin 放到前面，并补一条使用整数 cv 的回归测试。"
)

# ---------- 7 局限 ----------
h("7  局限", 1)
para(
    "（1）仅使用四个规模较小的内置数据集（150–1797 样本），结论不能外推到真实工业数据；"
    "（2）未做超参数搜索，λ 由手工扫描确定，存在选择偏差；"
    "（3）单一种子，未进行多次重复实验，报告的标准差为折间方差；"
    "（4）RADDT 的训练开销比 CART 高 2–3 个数量级，本文未讨论其工程可行性；"
    "（5）未与 OCT、ORT 等精确全局优化方法对比。"
)

# ---------- 8 结论 ----------
h("8  未来工作", 1)
para(
    "（1）结构化稀疏：用 Group Lasso 让不同内部节点使用不同的特征子集，而不是全局共享一个稀疏模式；"
    "（2）把树深纳入可微搜索，避免人工设定；"
    "（3）在 OpenML 等更大规模的基准上验证，而不是四个内置数据集；"
    "（4）与 OCT、ORT 等基于混合整数规划的精确全局优化方法做正面对比。"
)

h("9  结论", 1)
para(
    "本文复现了基于 ReLU+Argmin 重构的可微分决策树 RADDT，并在其分裂权重上引入 L1 稀疏正则。"
    "修正树深设置后，RADDT 在四个数据集上均优于 CART，并在两个数据集上追平随机森林；"
    "L1 正则的收益强依赖于强度选择，λ=1e-3 可在保持精度的同时获得约 7% 的权重稀疏。"
    "与结果相比，本文更想强调的是复现过程本身：三个不会报错的实现陷阱，足以让一整张实验表格失去意义。"
)

# ---------- 参考文献 ----------
h("参考文献", 1)
refs = [
    "[1] Q. Mao, J. Ren, Y. Wang, C. Zou, J. Zheng, Y. Cao. Differentiable Decision Tree via “ReLU+Argmin” "
    "Reformulation. NeurIPS 2025 (Spotlight).",
    "[2] L. Breiman, J. Friedman, C. J. Stone, R. A. Olshen. Classification and Regression Trees. Chapman & Hall, 1984.",
    "[3] L. Breiman. Random Forests. Machine Learning, 45(1): 5–32, 2001.",
    "[4] R. Tibshirani. Regression Shrinkage and Selection via the Lasso. JRSS-B, 58(1): 267–288, 1996.",
    "[5] T. Chen, C. Guestrin. XGBoost: A Scalable Tree Boosting System. KDD 2016: 785–794.",
]
for r in refs:
    para(r, size=9)

out = os.path.join(os.path.dirname(__file__), "Sparse-RADDT-report.docx")
DOC.save(out)
print("saved", out)
