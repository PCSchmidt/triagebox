# Evaluation protocol

Written before any results exist, so that the measurements cannot quietly shape the definitions.

## Corpus and questions

- **Documents:** public, unclassified material only (for example US-government publications and other openly licensed text). Source and licence are recorded per document. Nothing from any employer or controlled source.
- **Answerable questions:** each has one or more gold documents and a short gold answer string (a name, number or phrase) that must appear in a correct answer. Short factual answers keep the containment test meaningful.
- **Unanswerable questions**, in three kinds, each a share of the set:
  1. *Absent*: the topic is not in the corpus at all.
  2. *Near-miss*: a real entity or event is in the corpus but the asked-about fact is not (for example a different date or location).
  3. *False premise*: the question assumes something the documents contradict.
- **Size:** at least 150 answerable and 100 unanswerable questions, hand-checked. If a model drafts questions, every item is read and the gold answer verified against the source before it is kept.
- **Split:** a development set for prompt and retrieval tuning and a locked test set used once per configuration. Tuning on the test set invalidates it.

## Metrics

Defined in `triagebox.metrics`; repeated here as the contract.

| Metric | Definition |
| --- | --- |
| Hallucination rate | Answered ÷ all unanswerable questions (refusing is correct) |
| False refusal rate | Refused ÷ all answerable questions |
| Accuracy | Answerable questions answered with text containing the gold answer (normalised) ÷ all answerable |
| Citation validity | Citations to retrieved passages ÷ all citations, over answers |
| Gold citation recall | Answered answerable questions citing a gold document ÷ answered answerable |
| Malformed rate | Replies not following the `ANSWER:` / `INSUFFICIENT_EVIDENCE` protocol ÷ all |

Malformed replies are counted as neither answers nor refusals. Report them separately; a model that is "safe" only because it never follows the format has not shown anything.

**Faithfulness beyond citation validity** is checked with a judge model on a sample, and a hand-checked subset of the judge's verdicts is reported with the agreement rate. Until then, only citation validity and the lexical-support proxy are available, and results say so.

## Configurations

Model family at two or more sizes; precision FP16, INT8, INT4 (INT8 and INT4 by the schemes each runtime supports, named in the results); at least two runtimes where they run on the hardware. Fixed decoding settings (temperature 0), fixed prompt, fixed retrieval (top-k) across all configurations so differences come from model and precision only. Retrieval is varied in its own experiment.

## Measurement

- Time to first token and decode rate from a streamed generation; energy per answer as measured average board power × duration, never an assumed power figure.
- Each configuration is measured in several **interleaved rounds** on an otherwise idle machine, and compared within a round. A throttling GPU drifts during a run, and a single reading once misled a result in a sibling project.
- Report medians with ranges, the number of rounds, the hardware, the software versions and the date.

## Reporting

One table per experiment and one quality-versus-joules chart. Every result states what it does not show: single model family, single run unless repeated, the corpus size, and that desktop-GPU numbers are not edge-device numbers.
