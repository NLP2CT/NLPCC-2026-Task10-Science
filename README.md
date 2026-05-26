<div align="center">

<h1>NLPCC 2026 Shared Task 10: Reliability of AI-Assisted Scientific Reporting</h1>
<p><em>Two complementary tracks: claim-level faithfulness to experimental results and citation-level faithfulness to external evidence.</em></p>

[![Guidelines](https://img.shields.io/badge/Guidelines-Available-blue?style=for-the-badge)](GUIDELINES.md) [![Phase 1 Platform](https://img.shields.io/badge/Phase_1_Platform-Codabench-green?style=for-the-badge)](https://www.codabench.org/competitions/16666/) [![Registration](https://img.shields.io/badge/Registration-Closed-lightgrey?style=for-the-badge)](mailto:nlp2ct.runzhe@gmail.com) [![Website](https://img.shields.io/badge/Website-Live-purple?style=for-the-badge)](https://nlp2ct.github.io/NLPCC-2026-Task10-Science/)

</div>

<div align="center">
  <p>
    <a href="#latest-news"><b>News</b></a> &bull;
    <a href="#quick-links"><b>Quick Links</b></a> &bull;
    <a href="#introduction"><b>Introduction</b></a> &bull;
    <a href="#tracks"><b>Tracks</b></a> &bull;
    <a href="#schedule"><b>Schedule</b></a> &bull;
    <a href="#organizer"><b>Organizer</b></a>
  </p>
</div>

[中文版本 (Chinese Version)](README_zh.md)

---

## Latest News

- **[2026/05/26]** Phase 1 is now open. Phase 1 test inputs ([`data/`](data/)), a baseline prompting kit ([`baseline_prompting/`](baseline_prompting/)), and the offline evaluation scripts ([`offline_eval/`](offline_eval/)) are released. Submit Phase 1 results on [Codabench](https://www.codabench.org/competitions/16666/).
- **[2026/04/15]** Task guidelines and train-dev data for both tracks released.
- **[2026/03/20]** Shared task announced. Registration is now open.

## Quick Links

| | Link |
|---|---|
| Task Guidelines (EN / 中文) | [GUIDELINES.md](GUIDELINES.md) / [GUIDELINES_ZH.md](GUIDELINES_ZH.md) |
| Released Data (Train-Dev + Phase 1 Test) | [`data/`](data/) |
| Phase 1 Platform | [Codabench competition 16666](https://www.codabench.org/competitions/16666/) |
| Registration Status | Closed |
| Registration Email | [nlp2ct.runzhe@gmail.com](mailto:nlp2ct.runzhe@gmail.com) |
| Task Website | [nlp2ct.github.io/NLPCC-2026-Task10-Science](https://nlp2ct.github.io/NLPCC-2026-Task10-Science/) |

> **Data usage notice:** All released task data is provided exclusively for NLPCC 2026 Shared Task 10 during the competition period. Credits and attribution belong to the organizers, and the data may not be redistributed, mirrored, republished, relabeled, or released under any other name before the competition ends. After the competition, all training and test data will be redistributed under an open license for scientific research use.

## Introduction

As generative AI and agentic AI become increasingly integrated into scientific workflows, they are now widely used to assist with scientific writing, including summarizing experimental results, drafting conclusions, and generating citation-supported statements.

Recent studies have shown that AI-assisted scientific reporting often overgeneralizes conclusions beyond what the source evidence justifies. This shared task focuses on the reporting layer of AI-assisted research and centers on the following question:

> **Given scientific evidence and an AI-generated scientific statement, can a system determine whether the statement faithfully reflects the evidence it summarizes or cites?**

## Tracks

| | Track 1 | Track 2 |
|---|---|---|
| **Focus** | Claim-level faithfulness to experimental results | Citation-level faithfulness to external evidence |
| **Input** | Evidence bundle + claim paragraph (segmented into sentences) | Atomic scientific claim + cited paper full text |
| **Output** | One label per sentence | One support label + ranked evidence paragraph IDs |
| **Metrics** | Sentence Macro-F1 + Paragraph Exact Match (PEM) | Label Macro-F1 + Joint@3 |

### Track 1: Claim-level faithfulness to experimental results

Track 1 evaluates whether an AI-generated claim faithfully represents the experimental evidence it is intended to summarize. Systems are provided with a compact evidence bundle and an AI-generated claim paragraph, which is segmented into individual sentences for evaluation.

Participants are required to assign a label to each sentence, indicating whether it is supported by the evidence or, if not, what type of unsupported reporting it contains.

### Track 2: Citation-level faithfulness to external evidence

Track 2 evaluates whether an AI-generated claim with an associated citation is genuinely supported by the cited paper. Systems are given an atomic AI-generated scientific claim and the full text of the cited paper in structured textual form.

They must determine whether the paper directly supports the claim, partially supports it, is only topically related without providing evidential support, or is entirely irrelevant. In addition, systems are required to submit a ranked list of evidence paragraph IDs.

See the [Task Guidelines](GUIDELINES.md) for full definitions, data format, and examples.

## Schedule

| Date | Event |
|---|---|
| March 20, 2026 | Shared task announcement; registration opens |
| April 15, 2026 | Release of task guidelines and train-dev data |
| May 25, 2026 | Registration deadline |
| May 26, 2026 | Phase 1 data, evaluation entry, and offline evaluation scripts released |
| June 11, 2026 | Hidden test data release, no labels **(Phase 2)** |
| June 20, 2026 | Result submission deadline (Phase 1 + Phase 2) |
| June 30, 2026 | Evaluation results released; call for system reports |

## Evaluation Kit & Naive Baseline Kit

We provide a lightweight backbone evaluation stack for Phase 1: the official offline evaluator in [`offline_eval/`](offline_eval/) and a reference single-turn prompting kit in [`baseline_prompting/`](baseline_prompting/). The table below reports naive prompting reference results on the released Phase 1 setup.

| Model | T1 Score | T1 Macro-F1 | T1 PEM | T2 Score | T2 Macro-F1 | T2 Joint@3 | Avg(T1,T2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.1 Pro | 10.35% | 19.17% | 1.54% | 38.97% | **46.02%** | 31.91% | 24.66% |
| GPT-5.4 | 16.35% | **26.04%** | 6.66% | **40.51%** | 41.65% | **39.38%** | **28.43%** |
| Qwen3.6-Plus | **18.90%** | 25.33% | **12.46%** | 30.35% | 38.08% | 22.63% | 24.62% |

These numbers are intended as reproducible reference points for the released prompting kit, not as optimized leaderboard baselines.

## Organizer

- University of Macau

## Contacts

- Runzhe Zhan (Contact) | [Homepage](https://runzhe.me/)
- Derek F. Wong (Advisor) | [Homepage](https://www.fst.um.edu.mo/people/derekfw/)
- Yutong Yao
- Junchao Wu | [Homepage](https://junchaoiu.github.io/)
- Jingkun Ma
- Yanming Sun
- Fengying Ye
