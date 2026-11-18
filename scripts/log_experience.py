"""Reviewed log classification experiences; SQLite or PostgreSQL, no automatic training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "network": "DNS 조회 실패, 접속 시간 초과 등 네트워크 문제",
    "dependency": "패키지 버전 충돌이나 누락된 의존성",
    "compile": "소스 코드 구문 또는 타입 오류",
    "test": "테스트 assertion 실패",
    "other": "위 범주에 해당하지 않는 오류",
}


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, url, **connect_options):
        self.pg = url.startswith(("postgresql://", "postgres://"))
        if self.pg:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError:
                raise ValueError("PostgreSQL requires: pip install -r scripts/requirements-log-experience.txt") from None
            self.conn = psycopg.connect(url, row_factory=dict_row, autocommit=True, **connect_options)
        else:
            path = url.removeprefix("sqlite:///")
            if "://" in path:
                raise ValueError("Use a file path, sqlite:///path, or postgresql:// URL")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
        try:
            self.run("""CREATE TABLE IF NOT EXISTS log_experiences (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                input_sha256 TEXT NOT NULL, log_text TEXT NOT NULL,
                request_json TEXT NOT NULL, output_json TEXT NOT NULL,
                selected TEXT, raw_top1 TEXT NOT NULL,
                elapsed_ms DOUBLE PRECISION NOT NULL
            )""")
            self.run("""CREATE TABLE IF NOT EXISTS log_feedback (
                id TEXT PRIMARY KEY, experience_id TEXT NOT NULL
                REFERENCES log_experiences(id), created_at TEXT NOT NULL,
                label TEXT NOT NULL, action_success INTEGER,
                note TEXT NOT NULL, supersedes TEXT REFERENCES log_feedback(id)
            )""")
            self.conn.commit()
        except Exception:
            self.conn.close()
            raise

    def run(self, sql, params=()):
        return self.conn.execute(sql.replace("?", "%s") if self.pg else sql, params)

    def close(self):
        self.conn.close()

    def transaction(self):
        return self.conn.transaction() if self.pg else self.conn

    def record(self, log, request, output, elapsed_ms):
        results = output["response"]["results"]
        if len(results) != 1 or results[0]["id"] != "failure_category":
            raise ValueError("Unexpected classification response")
        result = results[0]
        selected = result["value"]["selected"]
        scores = result["scores"]
        if {s["id"] for s in scores} != set(LABELS) or len(scores) != len(LABELS):
            raise ValueError("Unexpected candidate labels")
        if selected is not None and selected not in LABELS:
            raise ValueError("Unexpected selected label")
        raw = max(scores, key=lambda s: s["option_probability"])["id"]
        identity = output["model"]
        if not identity.get("weights_sha256"):
            raise ValueError("Missing model identity")
        eid = str(uuid.uuid4())
        with self.transaction():
            self.run("INSERT INTO log_experiences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                eid, now(), hashlib.sha256(log.encode()).hexdigest(), log,
                json.dumps(request, ensure_ascii=False), json.dumps(output, ensure_ascii=False),
                selected, raw, elapsed_ms,
            ))
        return eid

    def review(self, eid, label, success, note):
        if label not in LABELS:
            raise ValueError("Unknown label")
        with self.transaction():
            if not self.pg:
                self.run("BEGIN IMMEDIATE")
            # Serialize feedback revisions per experience on PostgreSQL.
            suffix = " FOR UPDATE" if self.pg else ""
            if not self.run("SELECT id FROM log_experiences WHERE id = ?" + suffix, (eid,)).fetchone():
                raise ValueError("Unknown experience ID")
            previous = self.run("""SELECT f.id FROM log_feedback f
                WHERE experience_id = ? AND NOT EXISTS
                (SELECT 1 FROM log_feedback n WHERE n.supersedes = f.id)""", (eid,)).fetchone()
            fid = str(uuid.uuid4())
            self.run("INSERT INTO log_feedback VALUES (?, ?, ?, ?, ?, ?, ?)", (
                fid, eid, now(), label, success, note, previous["id"] if previous else None,
            ))
        return fid

    def rows(self):
        return [dict(r) for r in self.run("""SELECT e.*, f.label, f.action_success,
            f.note, f.id AS feedback_id FROM log_experiences e
            LEFT JOIN log_feedback f ON f.experience_id = e.id AND NOT EXISTS
            (SELECT 1 FROM log_feedback n WHERE n.supersedes = f.id)
            ORDER BY e.created_at, e.id""").fetchall()]


@contextmanager
def opened(url):
    store = Store(url)
    try:
        yield store
    finally:
        store.close()


def request_for(log):
    return {
        "state": {"log": log},
        "decisions": [{"id": "failure_category", "instruction": "로그에 나타난 실패 원인을 분류하세요.",
            "kind": {"type": "choice", "options": [
                {"id": key, "criterion": value} for key, value in LABELS.items()
            ]}}],
    }


def evaluate(rows):
    reviewed = [r for r in rows if r["label"] is not None]
    accepted = [r for r in reviewed if r["selected"] is not None]
    ratio = lambda n, d: n / d if d else None
    confusion = {}
    for r in reviewed:
        truth = confusion.setdefault(r["label"], {})
        predicted = r["selected"] or "abstained"
        truth[predicted] = truth.get(predicted, 0) + 1
    actions = [r for r in reviewed if r["action_success"] is not None]
    return {
        "total": len(rows), "reviewed": len(reviewed), "pending": len(rows) - len(reviewed),
        "unique_reviewed_inputs": len({r["input_sha256"] for r in reviewed}),
        "raw_top1_accuracy": ratio(sum(r["raw_top1"] == r["label"] for r in reviewed), len(reviewed)),
        "coverage": ratio(len(accepted), len(reviewed)),
        "abstention_rate": ratio(len(reviewed) - len(accepted), len(reviewed)),
        "accepted_accuracy": ratio(sum(r["selected"] == r["label"] for r in accepted), len(accepted)),
        "action_observations": len(actions),
        "action_success_rate": ratio(sum(r["action_success"] for r in actions), len(actions)),
        "confusion_truth_to_prediction": confusion,
        "warning": "Observational feedback; action success is not classification correctness. Repeats are not independent samples.",
    }


def export_rows(rows):
    for r in rows:
        if r["label"] is None:
            continue
        # Identical logs always go to the same partition, including across models.
        split = "validation" if int(r["input_sha256"][:8], 16) % 5 == 0 else "train"
        yield {
            "id": r["id"], "input_sha256": r["input_sha256"], "split": split,
            "request": json.loads(r["request_json"]), "target": r["label"],
            "classification_reward": None if r["selected"] is None else int(r["selected"] == r["label"]),
            "action_success": r["action_success"], "feedback_id": r["feedback_id"],
            "output": json.loads(r["output_json"]),
        }


def probability(value):
    x = float(value)
    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError("Expected a value between 0 and 1")
    return x


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("L2S1_EXPERIENCE_DB", str(ROOT / "results/log-experience.sqlite3")))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    classify = commands.add_parser("classify")
    classify.add_argument("--log-file", type=Path, required=True)
    classify.add_argument("--model", type=Path, default=ROOT / "models/Qwen3-4B-Q4_K_M.gguf")
    classify.add_argument("--binary", type=Path, default=ROOT / "target/release/l2s1")
    classify.add_argument("--threads", type=int, default=8)
    classify.add_argument("--timeout", type=float, default=180)
    classify.add_argument("--min-probability", type=probability, default=0.8)
    classify.add_argument("--min-mass", type=probability, default=0.05)
    review = commands.add_parser("review")
    review.add_argument("id")
    review.add_argument("--label", choices=LABELS, required=True)
    review.add_argument("--action-success", choices=["yes", "no", "unknown"], default="unknown")
    review.add_argument("--note", default="")
    listing = commands.add_parser("list")
    listing.add_argument("--pending", action="store_true")
    evaluate_cmd = commands.add_parser("evaluate")
    evaluate_cmd.add_argument("--model-sha256", help="Restrict metrics to one model")
    export = commands.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with opened(args.db) as store:
        if args.command == "classify":
            log = args.log_file.read_text(encoding="utf-8")
            if not log.strip():
                raise ValueError("Log must not be blank")
            if args.threads < 1 or args.timeout <= 0:
                raise ValueError("Threads and timeout must be positive")
            request = request_for(log)
            start = time.monotonic()
            process = subprocess.run([
                str(args.binary.resolve()), "--model", str(args.model.resolve()),
                "--device", "cpu", "--threads", str(args.threads), "--input", "-", "--diagnostics",
                "--min-top-probability", str(args.min_probability), "--min-candidate-mass", str(args.min_mass),
            ], input=json.dumps(request), text=True, capture_output=True, timeout=args.timeout)
            if process.returncode:
                raise ValueError("Inference failed; no experience saved. " + process.stderr[-2000:])
            output = json.loads(process.stdout)
            eid = store.record(log, request, output, (time.monotonic() - start) * 1000)
            print(json.dumps({"id": eid, "result": output["response"]["results"][0],
                "policy": output["response"]["policy"]}, ensure_ascii=False, indent=2))
        elif args.command == "review":
            fid = store.review(args.id, args.label, {"yes": 1, "no": 0, "unknown": None}[args.action_success], args.note)
            print(json.dumps({"feedback_id": fid}))
        elif args.command == "init":
            print(json.dumps({"initialized": True, "backend": "postgresql" if store.pg else "sqlite"}))
        else:
            rows = store.rows()
            if args.command == "list":
                print(json.dumps([{k: r[k] for k in ["id", "created_at", "selected", "label", "elapsed_ms"]}
                    for r in rows if not args.pending or r["label"] is None], indent=2))
            elif args.command == "evaluate":
                if args.model_sha256:
                    rows = [r for r in rows if json.loads(r["output_json"])["model"]["weights_sha256"] == args.model_sha256]
                print(json.dumps(evaluate(rows), ensure_ascii=False, indent=2))
            else:
                records = list(export_rows(rows))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as out:
                    for record in records:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps({"exported": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
