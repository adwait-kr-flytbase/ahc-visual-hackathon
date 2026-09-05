# Research agenda — open questions

Deliberately **questions, not answers.** No approach has been chosen yet.
Ordered by how much the answer would change what gets built.

> **Updated after the dataset doc.** Items marked ~~struck~~ are answered; **[NEW]** items
> were created by the dataset doc. See [`01-dataset.md`](01-dataset.md).

---

## 0. Now the highest-priority unknowns  **[NEW]**

0.1 **What are levels 1, 2, 3?** The dataset doc says `level` is "the task tier (below)" and then
    never defines the tiers. Level 1 has empty timestamps, 2 and 3 populated — so 1 is
    video-level and 2 is localisation, but **3 is a total unknown**. Everything about the
    output shape depends on this. Resolve from the CSVs, or ask.

0.2 **What is the submission format and the scoring function?** A *private evaluation system*
    exists. Event-level F1? Temporal IoU? mAP@tIoU? Frame-AUC? The metric determines whether
    to optimise for precision, for boundary accuracy, or for coverage. **Unknown.**

0.3 **Are all three levels scored, or do you pick one?** Is Level 3 worth more? Can you submit
    Level 1 only? This decides whether temporal localisation is optional or mandatory.

0.4 **Does the real-time constraint enter the score at all**, or is it judged separately/by
    inspection? "Runs in real time on limited GPU" is a stated constraint but a private
    scoring system can't easily measure it. If it's unscored, it's still a demo requirement.

---

## A. The problem shape

1. **What is the actual unit of decision?** A frame? A 1s clip? A 16-frame window? A tracked object
   over time? A scene-level state? The four temporal shapes (instant / gradual / duration-defined /
   static) may not share one unit — does the system need more than one?
2. **Is this detection or is it monitoring?** "Alert once when X starts" is a different system from
   "score every frame". The benchmark literature does the latter; the stated product need is the former.
3. **What does an alert contain** for it to be actionable? Class label only? Bounding box?
   Natural-language description? Confidence? Timestamp + geolocation?
4. **What's the ground truth for "real time"?** FPS on what hardware? Is 1 decision/second enough,
   or must every frame be scored? A drone at altitude may not need 30 Hz.

## B. The viewpoint gap

**Reframed:** drone is one of *three* viewpoints, not the primary one. CCTV and dashcam are
equally represented. So this is a **multi-domain** problem, and the fixed-camera literature is
more applicable than first assumed.

5. Nearly all VAD benchmarks (UCF-Crime, XD-Violence, ShanghaiTech, UBnormal, Avenue) are
   **fixed-camera CCTV**. Here the camera **moves and is overhead**. How much does that break:
   - background modelling / frame differencing?
   - "stationary object" detection (ego-motion vs. object motion)?
   - object scale — are cars 20px wide at altitude?
6. Provided data spans **drone, CCTV and dashcam** — all three are in scope for scoring.
   Is the camera domain *labelled* anywhere, or must it be inferred? Does one model handle all
   three, or does viewpoint-conditioned prompting / routing help?
7. **Night flights** are explicitly included. Do the candidate small VLMs work at all on
   low-light aerial IR/visible footage, or is that a separate model?

## C. Model selection

8. What is the **smallest VLM that still has the contextual reasoning**? Where does it break?
   Candidates implied by the tooling doc: Qwen3-VL-4B, Gemma 3 4B, Qwen2.5-VL 7B, Qwen3-VL 8B.
9. **CLIP-family vs. generative VLM** for the always-on stage. Alert-CLIP suggests CLIP-tuning is
   viable and it's orders of magnitude cheaper. What does a generative VLM buy that a tuned
   CLIP embedding + text queries doesn't?
10. **Does video-native input matter**, or is frame-sampling + a temporal aggregator enough?
    (`FPS_MAX_FRAMES` being called a latency dial suggests frames-per-inference is the main cost driver.)
11. What actually runs in real time on a **T4**? That's the free-tier hardware and probably the
    honest definition of "limited GPU capability" here. Need measured numbers, not estimates.

## D. Data and labels

12. ~~Footage is **unannotated**. What's the cheapest path to a usable training set?~~
    **ANSWERED: it is annotated** — class + temporal bounds + `description_summary`.
    Pseudo-labelling is now optional, not the bottleneck. Remaining question: is
    `description_summary` rich/consistent enough to SFT on, and how often is it blank?
13. **Anomalies are rare by definition.** How much of the provided footage contains any event at all?
    If it's <1%, mining positives is the bottleneck, not training.
14. ~~Is there a **held-out eval set** to trust?~~ **ANSWERED: yes** — public test, 34 videos,
    ~56 min, with ground truth, source-separated from train. Use it to validate the scoring
    pipeline *first*, before any modelling.
15. Can *additional* public benchmarks be used for training alongside the provided train split —
    and is that permitted by the rules? (Provided data is already curated from public sources,
    so overlap with the private eval set is a real leakage risk. Source separation is only
    guaranteed *within* the provided pack.)

## E. The false-positive problem

**Reframed:** with a fixed metric and a `normal` class in the taxonomy, false positives are now
*measurable* — the `normal` folder is the negative set. The question shifts from "how do we
avoid spam" to "what does the scoring function actually penalise".

16. What is the **normal-activity distribution** in this footage, and what does the system
    currently fire on that it shouldn't?
17. Is the right mechanism a threshold, a **cascade** (cheap stage recall-oriented, verifier
    precision-oriented), temporal persistence (fire only if N of M windows agree), or all three?
18. How do you **de-duplicate** an alert for a persistent condition (open drain) so it fires once,
    not forever? Event lifecycle: open → ongoing → closed.

## F. Evaluation & demo

19. What metric convinces a judge in a 5-minute demo? Frame-AUC (the literature standard) is
    probably *not* it — it doesn't capture "did it alert in time" or "did it spam".
20. Is there a better framing: **time-to-alert**, **alerts per hour of normal footage**,
    **event-level precision/recall**?
21. What does the demo actually look like — video playing with overlaid live alerts? That's likely
    the judgeable artifact, and it constrains the architecture (needs to run on a laptop/T4 live).

---

## G. Class-specific questions  **[NEW]**

22. `fire` and `smoke` are **separate classes** but co-occur constantly. How are overlapping
    events represented — two rows on the same interval? Does the metric penalise predicting both?
23. `loitering_or_suspicious_presence` is the only genuinely subjective class. What does the
    annotation actually look like, and how consistent is it?
24. `traffic_congestion` and `vehicle_blocking_traffic` and `stalled_or_broken_down_vehicle`
    are mutually confusable and causally linked (a stall causes a block causes congestion).
    Are they annotated as separate concurrent events on the same video?
25. Which of the 12 are **rare**? Class balance drives whether to weight the loss, oversample,
    or treat rare classes as a separate problem.
26. What is the **event duration distribution** per class — does it match the predicted
    instant / gradual / duration-defined / persistent split from `00-problem.md`?

## Things to go read (beyond the 4 given papers)
- The VAD benchmark landscape and what "anomaly" means in each
- Weakly-supervised VAD (MIL-based) — the dominant paradigm before VLMs
- Open-vocabulary / training-free VAD with CLIP
- Aerial/UAV-specific detection and tracking datasets (VisDrone, UAVDT) — viewpoint gap
- Small-VLM inference stacks: vLLM, SGLang, TensorRT-LLM, llama.cpp — what actually hits real time
- Quantisation for VLMs (AWQ/GPTQ/FP8) and its effect on visual reasoning quality
