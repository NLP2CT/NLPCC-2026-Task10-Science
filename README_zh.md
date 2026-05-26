<div align="center">

<h1>NLPCC 2026 共享任务 10：AI辅助科学报告的可靠性</h1>
<p><em>本任务设置两个互补赛道，分别关注实验结果的陈述级忠实性和外部文献的引文级忠实性。</em></p>

[![任务指南](https://img.shields.io/badge/%E4%BB%BB%E5%8A%A1%E6%8C%87%E5%8D%97-%E5%B7%B2%E5%8F%91%E5%B8%83-blue?style=for-the-badge)](GUIDELINES_ZH.md) [![第一阶段提交](https://img.shields.io/badge/%E7%AC%AC%E4%B8%80%E9%98%B6%E6%AE%B5%E6%8F%90%E4%BA%A4%E5%B9%B3%E5%8F%B0-Codabench-green?style=for-the-badge)](https://www.codabench.org/competitions/16666/) [![报名](https://img.shields.io/badge/%E6%8A%A5%E5%90%8D-%E5%B7%B2%E6%88%AA%E6%AD%A2-lightgrey?style=for-the-badge)](mailto:nlp2ct.runzhe@gmail.com) [![主页](https://img.shields.io/badge/%E4%B8%BB%E9%A1%B5-Live-purple?style=for-the-badge)](https://nlp2ct.github.io/NLPCC-2026-Task10-Science/)

</div>

<div align="center">
  <p>
    <a href="#最新动态"><b>最新动态</b></a> &bull;
    <a href="#快速入口"><b>快速入口</b></a> &bull;
    <a href="#简介"><b>简介</b></a> &bull;
    <a href="#赛道概览"><b>赛道</b></a> &bull;
    <a href="#日程安排"><b>日程</b></a> &bull;
    <a href="#主办单位"><b>主办</b></a>
  </p>
</div>

[English Version](README.md)

---

## 最新动态

- **[2026/05/26]** 第一阶段评测正式开启。第一阶段测试输入（[`data/`](data/)）、baseline prompting 套件（[`baseline_prompting/`](baseline_prompting/)）与离线评测脚本（[`offline_eval/`](offline_eval/)）已发布。请在 [Codabench](https://www.codabench.org/competitions/16666/) 提交第一阶段结果。
- **[2026/04/15]** 发布任务指南及两个赛道的训练开发数据。
- **[2026/03/20]** 共享任务公告发布，开放报名。

## 快速入口

| | 链接 |
|---|---|
| 任务指南（中文 / EN） | [GUIDELINES_ZH.md](GUIDELINES_ZH.md) / [GUIDELINES.md](GUIDELINES.md) |
| 已发布数据（训练开发 + 第一阶段测试） | [`data/`](data/) |
| 第一阶段提交平台 | [Codabench competition 16666](https://www.codabench.org/competitions/16666/) |
| 报名状态 | 已截止 |
| 报名邮箱 | [nlp2ct.runzhe@gmail.com](mailto:nlp2ct.runzhe@gmail.com) |
| 任务主页 | [nlp2ct.github.io/NLPCC-2026-Task10-Science](https://nlp2ct.github.io/NLPCC-2026-Task10-Science/) |

> **数据使用说明：**比赛期间，所有已发布的任务数据仅限用于参加 NLPCC 2026 Shared Task 10，相关署名权及发布权均归组织方所有；未经许可，不得以其他名义对数据进行二次分发、镜像发布、重新发布、重新命名或另行发布。比赛结束后，所有训练和测试数据将以开放协议重新分发，并可用于科研用途。

## 简介

随着生成式人工智能和智能体人工智能不断融入科学发现和研究流程，相关技术已被广泛用于辅助科学写作，例如总结实验结果、撰写研究结论，以及生成带有引文支持的科学表述。

然而，近期研究表明，AI 生成的科学报告常常会对结论作出超出原始证据支持范围的泛化表述。本共享任务聚焦于 AI 辅助科研中的"报告层"问题，围绕以下核心问题展开：

> **给定科学证据与一条由 AI 生成的科学陈述，系统能否判断该陈述是否忠实反映了其所概括或引用的证据？**

## 赛道概览

| | Track 1 | Track 2 |
|---|---|---|
| **方向** | 面向实验结果的陈述级忠实性判定 | 面向外部证据的引文级忠实性判定 |
| **输入** | 证据材料 + 陈述段落（按句切分） | 原子级科学陈述 + 被引论文全文 |
| **输出** | 每句一个标签 | 一个支持关系标签 + 排序的证据段落 ID |
| **指标** | 句子 Macro-F1 + 段落完全匹配（PEM） | 标签 Macro-F1 + Joint@3 |

### Track 1：面向实验结果的陈述级忠实性判定

本赛道旨在评估 AI 生成的科学陈述是否忠实表达了其所概括的实验性证据。参赛系统将获得一个紧凑的证据集合，以及一段由 AI 生成的陈述性段落。该段落将被切分为若干独立句子，参赛者需要对每个句子进行标注，判断其是否得到证据支持；若不被支持，则需进一步指出其所属的失实表述类型。

### Track 2：面向外部证据的引文级忠实性判定

本赛道旨在评估带有引文的 AI 生成科学陈述是否真正得到了其所引用论文的支持。参赛系统将获得一条原子化的 AI 生成科学陈述，以及被引用论文的结构化全文文本。

系统需要判断该论文与陈述之间的关系属于以下哪一类：直接支持、部分支持、仅主题相关但不构成证据支持、完全无关。此外，参赛系统还需提交一个按相关性排序的证据段落编号列表。

完整赛道定义、数据格式及示例详见[任务指南](GUIDELINES_ZH.md)。

## 日程安排

| 日期 | 事项 |
|---|---|
| 2026 年 3 月 20 日 | 共享任务公告发布；开放报名 |
| 2026 年 4 月 15 日 | 发布任务指南与训练开发数据 |
| 2026 年 5 月 25 日 | 报名截止 |
| 2026 年 5 月 26 日 | 发布第一阶段数据、评测入口和离线评测脚本 |
| 2026 年 6 月 11 日 | 发布隐藏测试数据（不含标签）**（第二阶段）** |
| 2026 年 6 月 20 日 | 结果提交截止日（第一阶段 + 第二阶段） |
| 2026 年 6 月 30 日 | 公布评测结果；征集系统报告 |

## 离线评测套件与朴素Prompting范例

我们为第一阶段提供了一个轻量级 backbone 评测栈：[`offline_eval/`](offline_eval/) 中的官方离线评测脚本，以及 [`baseline_prompting/`](baseline_prompting/) 中的参考 single-turn prompting kit。下表给出 naive prompting 策略在已发布第一阶段设置上的参考结果。

| 模型 | T1 Score | T1 Macro-F1 | T1 PEM | T2 Score | T2 Macro-F1 | T2 Joint@3 | Avg(T1,T2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.1 Pro | 10.35% | 19.17% | 1.54% | 38.97% | **46.02%** | 31.91% | 24.66% |
| GPT-5.4 | 16.35% | **26.04%** | 6.66% | **40.51%** | 41.65% | **39.38%** | **28.43%** |
| Qwen3.6-Plus | **18.90%** | 25.33% | **12.46%** | 30.35% | 38.08% | 22.63% | 24.62% |

这些分数用于为已发布的 prompting kit 提供可复现的参考点，并非经过优化的排行榜 baseline。


## 主办单位

- 澳门大学

## 联系人

- 詹润哲（Contact） | [Homepage](https://runzhe.me/)
- 黄辉（Advisor） | [Homepage](https://www.fst.um.edu.mo/people/derekfw/)
- 姚宇同
- 吴俊潮 | [Homepage](https://junchaoiu.github.io/)
- 马景坤
- 孙燕明
- 叶沣颖
