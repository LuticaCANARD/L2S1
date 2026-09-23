#!/usr/bin/env python3
"""Fixed-budget Gemma decision LoRA pilot. Not an implementation of proprietary RLCD.

Consumes production-tokenized inputs. No prompt reconstruction or generated text.
Only the train split enters optimization; calibration and test are evaluation-only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

MODEL = 'google/gemma-4-E2B-it'
REVISION = '3e22461f65e89153144f8adb70e3b8c2cc9845a7'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(row, logits, elapsed):
    import torch
    logits = logits.detach().float().cpu()
    candidates = logits[row['candidate_ids']]
    mass = torch.exp(torch.logsumexp(candidates,0)-torch.logsumexp(logits,0)).item()
    probs = candidates.softmax(0).tolist()
    scores = [dict(id=label,code=chr(65+i),raw_logit=z,option_probability=p)
              for i,(label,z,p) in enumerate(zip(row['option_ids'], candidates.tolist(), probs))]
    return dict(id=row['id'],elapsed_ms=elapsed,response=dict(results=[dict(id='airline_sentiment',scores=scores,
                candidate_mass=mass,truncated=False,input_tokens=len(row['input_ids']))]))


def evaluate(model, rows, path):
    import torch
    model.eval()
    with torch.inference_mode(), path.open('x') as output:
        # Untimed warmup; no labels are accessed by this function.
        model(input_ids=torch.tensor([rows[0]['input_ids']],device='cuda'),use_cache=False,logits_to_keep=1)
        for i,row in enumerate(rows):
            ids=torch.tensor([row['input_ids']],device='cuda')
            torch.cuda.synchronize();start=time.monotonic()
            logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1]
            torch.cuda.synchronize();elapsed=(time.monotonic()-start)*1000
            output.write(json.dumps(record(row,logits,elapsed))+'\n');output.flush()
            if (i+1)%100==0: print('EVAL',path.name,i+1,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--download-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    from huggingface_hub import snapshot_download
    checkpoint=snapshot_download(MODEL,revision=REVISION,cache_dir=args.cache,
                                 allow_patterns=['*.json','*.safetensors','*.jinja'],max_workers=2)
    print('CHECKPOINT',checkpoint,flush=True)
    if args.download_only:return
    import torch
    import transformers
    import peft
    from transformers import BitsAndBytesConfig, Gemma4ForConditionalGeneration, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    random.seed(20260923);torch.manual_seed(20260923);torch.cuda.manual_seed_all(20260923)
    manifest=json.loads((args.data/'selection.json').read_text());config=manifest['protocol']
    rows={name:read_jsonl(args.data/(name+'-tokens.jsonl')) for name in ['train','calibration','test','probe']}
    seen=set(manifest['excluded_prior_ids'])
    for split in ['train','calibration','test']:
        ids=set(manifest['splits'][split]['unique_ids'])
        assert not ids & seen, 'Training/evaluation/prior-example overlap'
        seen.update(ids)
    for name,values in rows.items():
        assert digest(args.data/(name+'.jsonl'))==manifest['splits'][name]['sha256']
        assert digest(args.data/(name+'-tokens.jsonl'))==manifest['splits'][name]['token_sha256']
        assert [r['id'] for r in values]==manifest['splits'][name]['ids']
        assert max(len(r['input_ids']) for r in values)<=2048
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    for row in rows['train']:
        assert [tokenizer.decode([i]) for i in row['candidate_ids']]==['A','B','C']
        assert len(set(row['candidate_ids']))==3
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16)
    model=Gemma4ForConditionalGeneration.from_pretrained(checkpoint,quantization_config=quant,
                    device_map={'':'cuda:0'},dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    # Freeze the base and prepare gradient checkpointing without PEFT's blanket
    # fp32 promotion of large embeddings (which would itself exceed this GPU).
    # Keep frozen matrices in bf16, normalization/scaling vectors in fp32.
    for name,param in model.named_parameters():
        param.requires_grad_(False)
        if param.is_floating_point() and param.ndim<2:
            param.data=param.data.float()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=config['rank'],lora_alpha=config['alpha'],
         lora_dropout=config['dropout'],bias='none',task_type='CAUSAL_LM',
         target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)'))
    trainable=[param for param in model.parameters() if param.requires_grad]
    metadata=dict(model=MODEL,revision=REVISION,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
        gpu=torch.cuda.get_device_name(),trainable_parameters=sum(x.numel() for x in trainable),protocol=config,
        token_sha256={n:digest(args.data/(n+'-tokens.jsonl')) for n in rows},script_sha256=digest(Path(__file__)),
        precision='NF4 base, bf16 non-norm frozen matrices, fp32 adapters/norms; SDPA',
        objective='candidate CE + 0.1 * negative log full-vocabulary candidate mass',smoke=args.smoke)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)
    if not args.smoke:
        for split in ['calibration','test','probe']:
            evaluate(model,rows[split],args.output/('base-'+split+'.jsonl'))
    # Exact same initial adapter (zero B matrices) used for baseline and learning.
    model.train()
    order=list(rows['train']);random.Random(20260923).shuffle(order)
    if args.smoke:order=order[:12]
    optimizer=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=0.0)
    accumulation=config['gradient_accumulation'];steps=math.ceil(len(order)/accumulation)
    optimizer.zero_grad(set_to_none=True);started=time.monotonic();sums=[0.,0.]
    with (args.output/'training.jsonl').open('x') as log:
        for i,row in enumerate(order):
            ids=torch.tensor([row['input_ids']],device='cuda')
            logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            candidate=logits[row['candidate_ids']]
            gold=row['option_ids'].index(manifest['labels'][row['id']])
            ce=torch.nn.functional.cross_entropy(candidate[None],torch.tensor([gold],device='cuda'))
            mass_loss=torch.logsumexp(logits,0)-torch.logsumexp(candidate,0)
            loss=ce+config['mass_loss_weight']*mass_loss
            assert torch.isfinite(loss), 'Nonfinite training loss'
            size=min(accumulation,len(order)-(i//accumulation)*accumulation)
            (loss/size).backward();sums[0]+=ce.item();sums[1]+=mass_loss.item()
            if (i+1)%accumulation==0 or i+1==len(order):
                grad=torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                step=(i//accumulation)+1
                optimizer.param_groups[0]['lr']=config['learning_rate']*(1-step/steps)
                info=dict(step=step,examples=i+1,ce=sums[0]/size,mass_loss=sums[1]/size,
                          grad_norm=grad.item(),elapsed_s=time.monotonic()-started,
                          peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
                log.write(json.dumps(info)+'\n');log.flush();sums=[0.,0.]
                print('TRAIN',json.dumps(info),flush=True)
    assert any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n)
    model.save_pretrained(args.output/'adapter');tokenizer.save_pretrained(args.output/'adapter')
    if args.smoke:
        evaluate(model,rows['train'][:3],args.output/'smoke-output.jsonl')
    else:
        model.gradient_checkpointing_disable()
        for split in ['calibration','test','probe']:
            evaluate(model,rows[split],args.output/('lora-'+split+'.jsonl'))
    (args.output/'complete.json').write_text(json.dumps(dict(steps=steps,training_and_final_evaluation_seconds=time.monotonic()-started))+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
