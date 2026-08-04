import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|> token in Qwen2-VL

_model = None
_processor = None

def load_local_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    print("[ATTENTION] Loading Qwen2-VL-2B locally (first time only)...")
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map = "auto",
        attn_implementation="eager"  # flash attention doesn't return attention weights
    )
    _processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=32*28*28,
        max_pixels=128*28*28
        )
    print("[ATTENTION] Model loaded.")
    return _model, _processor

def generate_attention_map(pil_image, question, save_path=None):
    model, processor = load_local_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"Look at the document carefully. Find the section relevant to this question: {question}\nAnswer with the exact text from the document. Be concise, one word or short phrase only."}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # generate the answer AND ask generate() to return attention weights for each step —
    # this avoids a separate second forward pass, which conflicts with Qwen2-VL's internal
    # rope_deltas caching if you try to re-run the model on the concatenated sequence yourself.
    # It's also cheaper: thanks to KV-caching, only the first generation step computes full
    # attention over the whole image; every later step is much smaller.
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True
        )

    gen_ids = gen_outputs.sequences

    raw_output = processor.batch_decode(
        gen_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    if "ANSWER:" in raw_output:
        answer = raw_output.split("ANSWER:")[-1].strip()
    else:
        answer = raw_output

    print(f"[ATTENTION] Answer: {answer}")

    # attention from the last generated token (end of the answer), last layer,
    # looking back over the full sequence (image + prompt + reasoning + answer so far)
    last_layer_attn = gen_outputs.attentions[-1][-1]   # (batch, heads, 1, seq_len_so_far)
    avg_attn = last_layer_attn[0].mean(dim=0)          # (1, seq_len_so_far)
    last_token_attn = avg_attn[0, :]                   # (seq_len_so_far,)

    # find image token positions in the full generated sequence
    orig_input_ids = gen_ids[0]
    image_positions = (orig_input_ids == IMAGE_PAD_TOKEN_ID).nonzero(as_tuple=True)[0]

    if len(image_positions) == 0:
        print("[ATTENTION] No image tokens found.")
        return answer, None

    image_attn = last_token_attn[image_positions].cpu().float().numpy()

    # reshape to spatial grid using image_grid_thw (t, h, w)
    grid_thw = inputs["image_grid_thw"][0]
    t, h, w = grid_thw.tolist()
    h_grid = h // 2  # qwen2-vl uses 2x2 patch merger
    w_grid = w // 2

    if len(image_attn) == t * h_grid * w_grid:
        attn_map = image_attn.reshape(h_grid, w_grid)
    else:
        side = int(np.sqrt(len(image_attn)))
        attn_map = image_attn[:side * side].reshape(side, side)

    # normalize and resize to original image size
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    orig_w, orig_h = pil_image.size
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.BILINEAR
        )
    ) / 255.0

    # figure: original | heatmap | overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(pil_image)
    axes[0].set_title("Input Document", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(attn_resized, cmap="hot")
    axes[1].set_title("Attention Map", fontsize=12)
    axes[1].axis("off")

    # escape literal $ so matplotlib doesn't try to parse it as math notation
    safe_question = question[:60].replace("$", "\\$")
    safe_answer = answer.replace("$", "\\$")

    axes[2].imshow(pil_image)
    axes[2].imshow(attn_resized, cmap="hot", alpha=0.5)
    axes[2].set_title(f"Q: {safe_question}\nA: {safe_answer}", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ATTENTION] Saved to {save_path}")

    return answer, fig
