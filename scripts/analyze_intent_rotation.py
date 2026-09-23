#!/usr/bin/env python3
"""Paired diagnostic: change display order/code assignment, not task semantics."""
import argparse
import collections
import json
from pathlib import Path

from evaluate_intents import load, rows, save, sha
from evaluate_intents_wide import code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root;out=root/'diagnostics'
    gold={r['id']:r for r in rows(root/'prepared/gold.jsonl') if r['dataset']=='banking77-en'}
    before={r['id']:r for r in rows(root/'runs/Qwen3-8B-Q8_0/predictions.jsonl') if r['id'] in gold}
    after_rows=rows(out/'qwen-rotation38.jsonl');after={r['id']:r for r in after_rows}
    assert len(after)==len(after_rows)==len(before)==len(gold)==200
    assert set(before)==set(after)==set(gold)
    full={r['id']:r for r in rows(root/'prepared/requests.jsonl')}
    diagnostic=rows(out/'banking77-requests.jsonl')
    assert diagnostic==[r for r in full.values() if r['id'] in gold]
    assert (out/'rotation38.exit-code').read_text().strip()=='0'
    counts=collections.Counter(); pairs=[]; lengths=collections.defaultdict(collections.Counter)
    for key in gold:
        p,q=before[key],after[key]
        assert not p.get('error') and not q.get('error')
        a,b=[r['response']['results'][0] for r in (p,q)]
        assert not a['truncated'] and not b['truncated']
        ab,bb=p['response']['backend'],q['response']['backend']
        assert ab.get('code_rotation',0)==0 and bb['code_rotation']==38
        for field in ('model_path','compute','offload_device','execution_mode','prompt_layout','prompt_profile'):
            assert ab[field]==bb[field]
        assert [s['id'] for s in a['scores']]==[s['id'] for s in b['scores']]
        assert len(a['scores'])==len(b['scores'])==77
        for i,s in enumerate(b['scores']):
            assert s['code']==code((i+77-38)%77,77)
        pick=lambda r:min(r['scores'],key=lambda s:(-s['option_probability'],s['id']))
        x,y=pick(a),pick(b)
        correct_a=x['id']==gold[key]['expected'];correct_b=y['id']==gold[key]['expected']
        counts['before_correct']+=correct_a;counts['after_correct']+=correct_b
        counts['prediction_changed']+=x['id']!=y['id']
        counts['correct_to_wrong']+=correct_a and not correct_b
        counts['wrong_to_correct']+=not correct_a and correct_b
        counts['wrong_both']+=not correct_a and not correct_b
        for label,result,chosen,correct in [('before',a,x,correct_a),('after',b,y,correct_b)]:
            if chosen['option_probability']>=.9:
                counts[label+'_confidence90_count']+=1
                counts[label+'_confidence90_wrong']+=not correct
        target_a=next(s for s in a['scores'] if s['id']==gold[key]['expected'])
        target_b=next(s for s in b['scores'] if s['id']==gold[key]['expected'])
        length_key=f"{len(target_a['token_ids'])}->{len(target_b['token_ids'])}"
        lengths[length_key]['n']+=1;lengths[length_key]['before_correct']+=correct_a;lengths[length_key]['after_correct']+=correct_b
        pairs.append(dict(id=key,expected=gold[key]['expected'],before=x['id'],after=y['id'],
            before_correct=correct_a,after_correct=correct_b,before_code=x['code'],after_code=y['code'],
            gold_code_before=target_a['code'],gold_code_after=target_b['code'],
            gold_token_lengths=[len(target_a['token_ids']),len(target_b['token_ids'])]))
    summary=dict(dataset='BANKING77 English',n=200,rotation=38,counts=dict(counts),
        gold_token_length_transitions={k:dict(v) for k,v in lengths.items()},
        requests_sha256=sha(out/'banking77-requests.jsonl'),predictions_sha256=sha(out/'qwen-rotation38.jsonl'),
        scope='Exploratory paired diagnostic; same text, candidate semantics and model. Display order and code assignment change together. Not a tuned replacement benchmark score.')
    save(out/'rotation-analysis.json',summary);save(out/'paired-predictions.json',pairs)
    base=load(out/'baseline-analysis.json')
    text=['# Intent classification diagnosis', '',
        'Qwen3-8B Q8_0, RTX 3060, frozen English/Korean 200-row test subsets. Baseline: English 111/200 (55.5%); Korean 98/200 (49.0%).', '',
        '## Findings from the original run', '',
        '- No inference errors, truncation, missing classes or missing rows were observed. The source-row/label mapping, complete candidate set and probability calculations were independently checked.',
        '- English had 89 errors. Only 14 examples had a gold label assigned a two-token code, of which 5 were correct. The remaining 186 examples had one-token gold codes and only 106 were correct. Multi-token gold codes therefore cannot account for most observed errors by themselves.',
        '- Korean had 102 errors. Twelve examples had two-token gold codes (4 correct); 188 had one-token gold codes (94 correct). These groups contain different labels and examples, so the rate difference is not a causal token-length estimate.',
        '- Candidate confidence is poorly calibrated as correctness: at confidence ≥90%, English had 58 errors among 163 predictions, and Korean had 68 among 160. These are candidate-relative probabilities, not calibrated correctness probabilities.',
        '- Examples of English confusion include `failed_transfer` → `declined_transfer` (6) and `compromised_card` → `contactless_not_working` (3). Korean confusions include `play_podcasts` → `play_music`, `email_query` → `email_sendemail` and `weather_query` → `datetime_query` (3 each). Some errors are fine-grained intent distinctions; others cross clearly different intents.', '',
        '## Paired display/code rotation control', '',
        'All original 200 English inputs and all 77 label meanings were retained. Rotation 38 changes which displayed code corresponds to each label and moves labels in the displayed list. This measures combined display-order/code-assignment sensitivity; it does not isolate tokenizer length from order or wording.', '',
        f"- Original accuracy: {counts['before_correct']}/200 ({counts['before_correct']/2:.1f}%). Rotated accuracy: {counts['after_correct']}/200 ({counts['after_correct']/2:.1f}%).",
        f"- Changed predictions: {counts['prediction_changed']}/200 ({counts['prediction_changed']/2:.1f}%). Correct → wrong: {counts['correct_to_wrong']}; wrong → correct: {counts['wrong_to_correct']}.",
        f"- Rotated predictions at confidence ≥90%: {counts['after_confidence90_wrong']} errors among {counts['after_confidence90_count']} predictions.", '',
        'This is a diagnosis on the test subset, not independent validation of a better prompt. The original benchmark scores remain unchanged. Any proposed prompt, code-bias correction or calibration should be chosen on separate development/calibration data before a new held-out evaluation.', '',
        '## Interpretation', '',
        'The current setup is zero-shot intent selection from short label-name criteria, using a general language model and answer-code likelihoods. It is not a classifier trained on BANKING77/MASSIVE examples. Label semantics, option presentation, code priors and the absence of calibration are plausible contributors. The measured control quantifies ordering/code sensitivity; it does not prove which factor causes every error. More model parameters or a larger candidate limit alone do not establish better intent classification.', '',
        'Next experiments should use a separate development split: explicit intent definitions and confusing-class examples; label/order robustness controls; then held-out calibration or task-specific training. Preserve the untouched test split for final validation.', '',
        'Evidence: `baseline-analysis.json`, `rotation-analysis.json`, `paired-predictions.json`, exact input/output JSONL, GPU log and `rotate.sh`.', '']
    (out/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
