# NLPCC 2026 Shared Task 10: The Reliability of AI-Assisted Scientific Reporting

**Task Guidelines**

The 15th CCF International Conference on Natural Language Processing and Chinese Computing (NLPCC 2026)

---

## 1. Track Definitions

### Track 1: Claim-Level Faithfulness to Experimental Results

Each example contains:

- **Evidence bundle**: a result paragraph, optionally with tables/charts, extracted from an open-access NLP paper.
- **Claim paragraph**: typically 2-4 sentences, generated to summarize or interpret the evidence bundle.

For each sentence in the paragraph, systems must output **one label** from the following set:

| Label | Description |
|---|---|
| `Supported` | The sentence is adequately supported by the evidence bundle. |
| `Unsupported Causal Mechanistic` | The sentence introduces unsupported causal or mechanistic interpretation. |
| `Unsupported Entity` | The sentence mentions unsupported datasets, metrics, model variants, baselines, or related scientific entities. |
| `Scope Overgeneralization` | The sentence extends the supported scope beyond what the evidence warrants. |
| `Contradiction` | The sentence directly conflicts with the evidence. |

Each sentence is assigned a single primary label. In rare cases, a sentence may be annotated with multiple applicable labels; predicting any one of them is considered correct for evaluation purposes.

#### Track 1 Train-Dev Data Format

Each record in the released JSONL file contains the following fields:

```json
{
  "claim_text": "string — the claim paragraph",
  "evidence_bundle": [
    {
      "type": "text",
      "text": "result paragraph text ..."
    },
    {
      "type": "table",
      "table_caption": ["Table 3: ..."],
      "img_path": "images/xxx.jpg"
    },
    {
      "type": "image",
      "image_caption": ["Figure 1: ..."],
      "img_path": "images/xxx.jpg"
    }
  ],
  "sentence_label": [
    {
      "sentence": "sentence text ...",
      "types": ["Supported"]
    },
    {
      "sentence": "sentence text ...",
      "types": ["Unsupported Entity"]
    }
  ]
}
```

**Field descriptions:**

| Field | Type | Description |
|---|---|---|
| `claim_text` | string | The full claim paragraph. |
| `evidence_bundle` | list of objects | One or more evidence items. Each item has a `type` field (`"text"`, `"table"`, or `"image"`) and type-specific content fields. |
| `sentence_label` | list of objects | Per-sentence annotations. Each object contains a `sentence` (string) and `types` (list of labels; typically one, see note above). |

> At test time, the `types` field in `sentence_label` will be withheld. Participants will receive `claim_text`, `evidence_bundle`, and the sentence segmentation.

The image files referenced by `img_path` in the evidence bundle are provided separately in `images.zip` (train-dev) and `images-testp1.zip` (Phase 1 test). Inside the archives, files are stored at the archive root (e.g. `<sha>.jpg`). To match the relative `images/<sha>.jpg` paths used in the JSONL records, extract each archive into a directory named `images/` (for example `unzip data/images.zip -d data/images/`).

#### Track 1 Train-Dev Data Statistics

| | Count |
|---|---|
| Records | 3,333 |
| Total sentences | 17,547 |

**Negative label distribution (counts are not mutually exclusive):**

| Label | Count |
|---|---|
| `Unsupported Entity` | 639 |
| `Scope Overgeneralization` | 605 |
| `Unsupported Causal Mechanistic` | 276 |
| `Contradiction` | 216 |

### Track 2: Citation-Level Faithfulness to External Evidence

Each example contains:

- **Atomic claim**: a single scientific claim associated with one cited paper.
- **Cited paper (full text)**: the cited paper in structured textual form, represented as a list of paragraphs with stable paragraph IDs (e.g., `P1`, `P2`, ...).

Systems must output:

1. **One support label** from the following set:

| Label | Description |
|---|---|
| `Supported` | The cited paper provides evidence directly supporting the claim. |
| `Overstate` | The claim overstates what the cited paper supports. |
| `Topical Match` | The cited paper is topically related but does not provide the evidential support required for the claim. |
| `Irrelevant` | The cited paper has no meaningful support relation to the claim. |

2. **A ranked list of up to *k* = 3 evidence paragraph IDs**, identifying the paragraphs in the cited paper that are most relevant to the support judgment.

#### Track 2 Train-Dev Data Format

Each record in the released Train-Dev data contains the following fields:

```json
{
  "claim_text": "string — the scientific claim",
  "cited_paper_full_text": [
    {"P1": "paragraph text ..."},
    {"P2": "paragraph text ..."},
    ...
  ],
  "evidence_para_ids": ["P12", "P13"],
  "evidence_texts": {
    "P12": "paragraph text ...",
    "P13": "paragraph text ..."
  },
  "label": "Supported | Overstate | Topical Match | Irrelevant"
}
```

**Field descriptions:**

| Field | Type | Description |
|---|---|---|
| `claim_text` | string | The atomic scientific claim to be verified. |
| `cited_paper_full_text` | list of objects | The full cited paper, represented as a list of `{paragraph_id: text}` objects in document order. |
| `evidence_para_ids` | list of strings | Gold paragraph IDs that serve as evidence for the support judgment. |
| `evidence_texts` | object | A convenience mapping from each gold evidence paragraph ID to its text content. |
| `label` | string | The gold support label (one of: `Supported`, `Overstate`, `Topical Match`, `Irrelevant`). |

> At test time, the `label`, `evidence_para_ids`, and `evidence_texts` fields will be withheld. Participants will receive only `claim_text` and `cited_paper_full_text`.

#### Track 2 Train-Dev Data Statistics

| Label | Count |
|---|---|
| `Supported` | 1,029 |
| `Irrelevant` | 382 |
| `Topical Match` | 376 |
| `Overstate` | 375 |
| **Total** | **2,162** |

---

## 2. Evaluation

### Metrics

**Track 1** uses two official metrics:

1. **Sentence-level Macro-F1** over all labels — balanced performance across all categories.
2. **Paragraph Exact Match (PEM)** — a paragraph is correct only if *all* sentence labels are predicted correctly.

**Track 1 ranking score** = (Macro-F1 + PEM) / 2

**Track 2** uses two official metrics:

1. **Label Macro-F1** over the four support labels — balanced performance on support-relation classification.
2. **Joint@3** — a prediction is counted as successful only if the system assigns the **correct support label** *and* retrieves **at least one correct evidence paragraph** in its top-3 evidence predictions.

**Track 2 ranking score** = (Macro-F1 + Joint@3) / 2

For Track 2, when the gold evidence list of a sample is empty, Joint@3 is
counted as successful only if the predicted label is correct *and* the
predicted evidence list is also empty (the `match-empty` policy of the
offline evaluator). This is the policy used for the official ranking on both
the public leaderboard (Phase 1) and the hidden test set (Phase 2).

### Two-Phase Evaluation

Evaluation is conducted in two phases to balance development flexibility with robustness against overfitting.

**Phase 1 — Open Evaluation (May 26 -- June 20, 2026):**
The Phase 1 test data, baseline prompting kit, and offline evaluation scripts have been released on May 26 (see the `data/`, `baseline_prompting/`, and `offline_eval/` directories in the repository). Participants must submit result files to the [Phase 1 Codabench platform](https://www.codabench.org/competitions/16666/) in the official format to obtain their Phase 1 scores. The public leaderboard will remain open until the final submission deadline.

**Phase 2 — Hidden Evaluation (June 11 -- June 20, 2026):**
On June 11, the organizers will release a previously unseen held-out test set (without labels). Participants must run their final models on this hidden test set and submit result files to the platform before the submission deadline (June 20).

**Final Score:**
The final ranking score is the average of the Phase 1 (open) and Phase 2 (hidden) scores. This two-phase design discourages overfitting to the open test data and better reflects system generalizability.

### Data Release and Usage Policy

All data released for this shared task, including train/dev data, open test data, and hidden test data, is provided exclusively for participation in NLPCC 2026 Shared Task 10 during the competition period.

During the competition period, all credits and attribution for the released data belong to the organizers. Participants and third parties may not redistribute, mirror, republish, relabel, or create alternative releases of the data under any other name.

After the competition concludes, all task data, including both training and test data, will be redistributed under an open license and may be used for scientific research purposes.

---

## 3. Submission Requirements

Each participating team must submit:

1. **Result files** in the official submission format, submitted to the platform for both Phase 1 (open) and Phase 2 (hidden) test inputs. The Phase 1 public platform is [Codabench competition 16666](https://www.codabench.org/competitions/16666/).

2. **A technical report** that comprehensively describes:
   - Training data sources and any data augmentation or preprocessing steps
   - Model architecture and design choices
   - Prompting strategy or retrieval setup (if applicable)
   - Training procedure and hyperparameters
   - Evaluation methodology and any ablation results

   The technical report must provide sufficient detail for full reproducibility. The organizers reserve the right to request **verification materials** — including source code, model weights, and training scripts — for any submission where reproducibility concerns arise.

---

## 4. Schedule

| Date | Event |
|---|---|
| March 20, 2026 | Shared task announcement and call for participation |
| March 20, 2026 | Registration opens |
| April 15, 2026 | Release of detailed task guidelines and training data |
| May 25, 2026 | Registration deadline |
| May 26, 2026 | Phase 1 data, evaluation entry, and offline evaluation scripts released |
| June 11, 2026 | Hidden (held-out) test data release, no labels **(Phase 2 begins)** |
| June 20, 2026 | Deadline for participants to submit all results (Phase 1 + Phase 2) |
| June 30, 2026 | Evaluation results released; call for system reports |

---

## 5. FAQ

**Q: Do teams have to participate in both tracks?**
- A: No. Teams may participate in either track or both. Tracks are evaluated independently.

**Q: How are awards determined?**
- A: We follow the NLPCC [official guidelines](http://tcci.ccf.org.cn/conference/2026/shared-tasks/). The top-ranked team in each track will receive a certificate jointly issued by NLPCC and CCF-NLP. We are also considering an award for the top-ranked team based on the average score across both tracks (to be confirmed).

**Q: Does each Track 1 sample contain only one piece of evidence?**
- A: No. An evidence bundle may contain multiple materials — for example, a result paragraph together with one or more paragraphs, tables or charts. Systems should consider all provided evidence when making predictions.

**Q: Track 1 may include multimodal inputs (e.g., tables or charts). Are participants required to use end-to-end multimodal models?**
- A: No. There are no restrictions on system design. Participants may use any approach they see fit.

**Q: In Track 2, what if a claim references multiple citations?**
- A: Each Track 2 data sample is paired with cited papers. Participants only need to consider the full-text material provided for that sample; the task will not require resolving references beyond the given cited paper.

**Q: Can we submit multiple times to the online platform during Phase 1 (open evaluation)?**
- A: Yes. Participants may submit multiple times during the open evaluation period. The leaderboard will reflect the latest submission.

**Q: Can we use external datasets or additional papers beyond the released train/dev data for training?**
- A: Yes. Training data sources are unrestricted. All external data sources used must be disclosed in the technical report. We will categorize and summarize the different approaches in the final task overview paper, but the choice of training data will not affect scoring & ranking.

**Q: Can we use proprietary LLMs (e.g., GPT-5, Claude) via API in our systems?**
- A: Yes. However, participants must clearly report the model name, version (w/ request date), and API costs in the technical report.
