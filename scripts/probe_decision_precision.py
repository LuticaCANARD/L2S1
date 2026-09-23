#!/usr/bin/env python3
"""Post-hoc inference-only precision diagnostic on 100 calibration inputs."""
import argparse
import json
from pathlib import Path
from train_decision_lora import MODEL, REVISION, evaluate, read_jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import Gemma4ForConditionalGeneration
    from peft import PeftModel
    torch.set_num_threads(4)
    checkpoint=snapshot_download(MODEL,revision=REVISION,cache_dir=a.root/'hf-cache',local_files_only=True,
                                 allow_patterns=['*.json','*.safetensors','*.jinja'])
    out=a.root/'bf16-probe';out.mkdir(exist_ok=False)
    rows=read_jsonl(a.root/'data/calibration-tokens.jsonl')[:100]
    model=Gemma4ForConditionalGeneration.from_pretrained(checkpoint,device_map={'':'cuda:0'},dtype=torch.bfloat16,attn_implementation='sdpa')
    for param in model.parameters():
        param.requires_grad_(False)
        if param.is_floating_point() and param.ndim<2:param.data=param.data.float()
    evaluate(model,rows,out/'base-calibration.jsonl')
    model=PeftModel.from_pretrained(model,a.root/'pilot/adapter',is_trainable=False)
    evaluate(model,rows,out/'lora-calibration.jsonl')
    (out/'complete.json').write_text(json.dumps(dict(cases=100,peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3,
        scope='Post-hoc BF16 inference diagnostic on first 100 calibration inputs. No additional training or tuning.'))+'\n')


if __name__=='__main__':main()
