# **Hackathon Primer — Small VLMs for Video Anomaly Detection**

**Where to start: prior work and fine-tuning tools examples.**

**1\. SOTA Session: [AHC\_VAD\_HACKATHON\_SOTA.pptx](https://docs.google.com/presentation/d/1PiYW8hE5h8UNtveXxIxGm1U4q47h4_76/edit?usp=sharing&ouid=110085763367645945404&rtpof=true&sd=true)**

## **2\. Good Reads \- VAD**

1. Alert-CLIP: Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly Detection :[Link](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_Alert-CLIP_Abnormality-aware_Latent-Enhanced_Representation_Tuning_of_CLIP_for_Video_Anomaly_CVPR_2026_paper.pdf)  
2. Similar Approach (uses fine-grained prompting): [Link](https://openaccess.thecvf.com/content/WACV2026/html/Zou_Unlocking_Vision-Language_Models_for_Video_Anomaly_Detection_via_Fine-Grained_Prompting_WACV_2026_paper.html)  
3. Cerberus: Real-Time Video Anomaly Detection via Cascaded Vision-Language Models: [Link](https://arxiv.org/html/2510.16290v)  
4. TAU-R1 (Traffic Anomaly Understanding):[Link](https://arxiv.org/abs/2603.19098)

## **3\. Example \- Fintuning Frameworks**

Following are some good resources & examples:

**Unsloth** — fastest start. Free Colab notebooks for Qwen3-VL 8B, Gemma 3 4B, Qwen2.5-VL 7B, Llama 3.2 Vision 11B, Pixtral 12B.

example:

| model \= FastVisionModel.get\_peft\_model(    model,    finetune\_vision\_layers   \= False,   \# frozen encoder    finetune\_language\_layers \= True,    r \= 16, lora\_alpha \= 16,    target\_modules \= "all-linear",) |
| :---- |

Tip: Use `UnslothVisionDataCollator` with `train_on_responses_only = True`. Build the dataset with a list comprehension, **not** `dataset.map()` — mapping breaks on multi-image samples. Refer the following doc: 📄 [unsloth.ai/docs/basics/vision-fine-tuning](http://unsloth.ai/docs/basics/vision-fine-tuning)

**ms-swift** — CLI-driven, broader model coverage, training \+ inference \+ eval \+ export in one tool.

example:

| swift sft \--model Qwen/Qwen3-VL-4B-Instruct \\    \--dataset train.jsonl \--val\_dataset val.jsonl \\    \--tuner\_type lora \--lora\_rank 8 \--lora\_alpha 32 \\    \--freeze\_vit true \--freeze\_aligner true \\    \--torch\_dtype bfloat16 \--learning\_rate 1e-4 \\    \--num\_train\_epochs 1 \--per\_device\_train\_batch\_size 1 \\    \--gradient\_accumulation\_steps 2 \--gradient\_checkpointing true \\    \--max\_length 4096 \--output\_dir output |
| :---- |

`Tip: IMAGE_MAX_TOKEN_NUM` and `FPS_MAX_FRAMES` are your memory and latency dials. 📄 [swift.readthedocs.io](https://swift.readthedocs.io/en/latest/) · [dataset formats](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html) · [Qwen3-VL guide](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html)

**HF TRL \+ PEFT** — the reference stack. Set `max_length=None` in `SFTConfig` or truncation silently cuts image tokens. 📄 [TRL VLM SFT](https://huggingface.co/docs/trl/main/en/training_vlm_sft) · [HF Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)

