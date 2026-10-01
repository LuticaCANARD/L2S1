#!/usr/bin/env python3
"""Compare our answer serializer with pinned, unmodified Ollaya Rust source."""
import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

from .prepare_jev_data import decision
from .report_jev import answer
from .common import digest, require, write_json

REVISION = 'f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea'


def main():
    p = argparse.ArgumentParser(prog="l2s1-train verify-ollaya", description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Existing Ollaya checkout at the pinned revision')
    p.add_argument('--output', type=Path, required=True, help='New scratch directory')
    a = p.parse_args()
    actual = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == REVISION, 'Wrong Ollaya reference revision')
    changes = subprocess.check_output(['git', '-C', str(a.source), 'diff', 'HEAD', '--', 'crates/ollaya-decision/src'], text=True)
    require(not changes, 'Ollaya reference decision source was modified')
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.source/'crates/ollaya-decision/src'
    shutil.copytree(source, a.output/'src')
    (a.output/'Cargo.toml').write_text('''[package]
name = "ollaya-answer-parity"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
indexmap = {version="=2.14.1",features=["serde"]}
serde = {version="1.0",features=["derive"]}
serde_json = {version="1.0.151",features=["preserve_order"]}
thiserror = "2.0.20"
''')
    (a.output/'src/main.rs').write_text('''use std::io::{self,BufRead};
use ollaya_answer_parity::{Answer,parse_questions};
fn main() {
 for line in io::stdin().lock().lines() {
  let v:serde_json::Value=serde_json::from_str(&line.unwrap()).unwrap();
  let qs=parse_questions(&v["questions"]).unwrap();
  let q=&qs["q"];
  let probabilities=serde_json::from_value(v["probabilities"].clone()).unwrap();
  let a=Answer {qtype:q.qtype,probabilities,act_probability:None};
  println!("{}",a.to_typesafe(q));
 }
}
''')
    subprocess.run(['cargo', 'build', '--offline', '--manifest-path', str(a.output/'Cargo.toml')], check=True)
    rng, records, expected = random.Random(20260927), [], []
    for kind in ('choice', 'noul', 'score'):
        for i in range(100):
            n = 2 if kind == 'noul' else 2+i%9
            ps = [1/n]*n if i%10 == 0 else [rng.random() for _ in range(n)]
            ps = [v/sum(ps) for v in ps]
            q = dict(type=kind, instructions='Evaluate this case')
            if kind == 'choice':
                q['criteria'] = {f'option-{j}':f'Criterion {j}' for j in range(n)}
            elif kind == 'score':
                q['criteria'] = [f'Level {j}' for j in range(n)]
            ids = ['false', 'true'] if kind == 'noul' else list(q['criteria']) if kind == 'choice' else list(map(str, range(n)))
            expected.append(answer(decision('q', q), dict(id='q', scores=[
                dict(id=k, option_probability=v) for k,v in zip(ids, ps)])))
            records.append(dict(questions={'q':q}, probabilities=ps))
    import os
    binary = 'ollaya-answer-parity.exe' if os.name == 'nt' else 'ollaya-answer-parity'
    run = subprocess.run([str((a.output/'target/debug'/binary).resolve())],
                         input=''.join(json.dumps(r)+'\n' for r in records), text=True, capture_output=True, check=True)
    actual = [json.loads(line) for line in run.stdout.splitlines()]
    require(actual == expected, 'Ollaya answer rendering differs')
    write_json(a.output/'parity.json', dict(reference_revision=REVISION,
        reference_answer_sha256=digest(source/'answer.rs'), cases=len(records), passed=len(actual),
        scope='Exact synthetic answer JSON parity only; no HTTP or model equivalence claim'))
    print(f'{len(actual)}/{len(records)} answer JSONs match')


if __name__ == '__main__':
    main()
