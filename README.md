<div align="center">

<h1>NLPCC 2026 Shared Task 10: Reliability of AI-Assisted Scientific Reporting</h1>
<p><em>Two complementary tracks: claim-level faithfulness to experimental results and citation-level faithfulness to external evidence.</em></p>

[![Guidelines](https://img.shields.io/badge/Guidelines-Available-blue?style=for-the-badge)](GUIDELINES.md) [![Data](https://img.shields.io/badge/Train--Dev_Data-Released-green?style=for-the-badge)](data/) [![Registration](https://img.shields.io/badge/Registration-Closed-lightgrey?style=for-the-badge)](mailto:nlp2ct.runzhe@gmail.com) [![Website](https://img.shields.io/badge/Website-Live-purple?style=for-the-badge)](https://nlp2ct.github.io/NLPCC-2026-Task10-Science/)

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

- **[2026/05/26]** Notice: Due to some last-minute registrations received on May 25, Phase 1 data, the evaluation entry, and the offline evaluation scripts will be released on May 26. We will notify all registered teams by email with further details ASAP.
- **[2026/04/15]** Task guidelines and train-dev data for both tracks released.
- **[2026/03/20]** Shared task announced. Registration is now open.

## Quick Links

| | Link |
|---|---|
| Task Guidelines (EN / 中文) | [GUIDELINES.md](GUIDELINES.md) / [GUIDELINES_ZH.md](GUIDELINES_ZH.md) |
| Train-Dev Data | [`data/`](data/) |
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
