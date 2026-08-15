# models/

Two files, two different models, two different jobs. Grouped here because both wrap a
Qwen2-VL variant, not because they're used together.

| File | Model | Where it's used | Why |
|---|---|---|---|
| `qwen.py` | Qwen2-VL-max, hosted via DashScope API | The core agent (`agent/nodes/extractor.py`) and most of `evaluation/` | This is "the model" for the project and  structured field extraction and document QA. |
| `attention_map.py` | Qwen2-VL-2B, loaded locally | `evaluation/attention_eval.py`, `evaluation/human_eval_attention_maps.py` | RQ3 explainability only. Needs a *local* model because it reads raw attention weights off the forward pass to render heatmaps, the hosted API doesn't expose those. Deliberately a smaller model (2B vs. the hosted max variant) to fit the project's GPU allocation. |

