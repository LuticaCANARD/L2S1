import os
import tempfile
import unittest
from pathlib import Path

from log_experience import LABELS, Store, evaluate, export_rows, request_for


def fixture(selected="network"):
    return {"model": {"weights_sha256": "fixture-model"}, "response": {"results": [{
        "id": "failure_category", "value": {"selected": selected},
        "scores": [{"id": label, "option_probability": 0.9 if label == "network" else 0.025}
                   for label in LABELS],
    }]}}


class StoreContract:
    def test_persistence_after_close(self):
        eid = self.store.record("persist", request_for("persist"), fixture(), 1)
        self.store.review(eid, "network", None, "")
        self.store.close()
        self.reopen()
        self.assertEqual(self.store.rows()[0]["label"], "network")

    def test_review_revision_and_separate_outcomes(self):
        eid = self.store.record("DNS failure", request_for("DNS failure"), fixture(), 12)
        self.assertIsNone(evaluate(self.store.rows())["raw_top1_accuracy"])
        first = self.store.review(eid, "compile", 1, "Initial review")
        self.assertEqual(evaluate(self.store.rows())["accepted_accuracy"], 0)
        second = self.store.review(eid, "network", 0, "Corrected diagnosis")
        metrics = evaluate(self.store.rows())
        self.assertEqual(metrics["reviewed"], 1)
        self.assertEqual(metrics["accepted_accuracy"], 1)
        self.assertEqual(metrics["action_success_rate"], 0)
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.run("SELECT COUNT(*) AS n FROM log_feedback").fetchone()["n"], 2)
        with self.assertRaises(ValueError):
            self.store.review("unknown", "network", None, "")

    def test_abstention_export_and_duplicate_partition(self):
        first = self.store.record("same log", request_for("same log"), fixture(None), 12)
        self.store.review(first, "network", None, "")
        second = self.store.record("same log", request_for("same log"), fixture(), 10)
        self.store.review(second, "network", None, "")
        self.store.record("unreviewed", request_for("unreviewed"), fixture(), 10)
        metrics = evaluate(self.store.rows())
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["raw_top1_accuracy"], 1)
        self.assertEqual(metrics["unique_reviewed_inputs"], 1)
        exported = list(export_rows(self.store.rows()))
        self.assertEqual(len(exported), 2)
        self.assertEqual(exported[0]["split"], exported[1]["split"])
        self.assertIsNone(exported[0]["classification_reward"])
        self.assertIsNone(exported[0]["action_success"])
        self.assertEqual(exported[0]["target"], "network")

    def test_reject_invalid_output_without_recording(self):
        output = fixture("invalid")
        with self.assertRaises(ValueError):
            self.store.record("log", request_for("log"), output, 10)
        self.assertEqual(self.store.rows(), [])


class SQLiteTests(StoreContract, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "experience.db")
        self.reopen()

    def reopen(self):
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()


@unittest.skipUnless(os.environ.get("L2S1_TEST_POSTGRES_URL"), "Set L2S1_TEST_POSTGRES_URL to a disposable database")
class PostgreSQLTests(StoreContract, unittest.TestCase):
    def setUp(self):
        import psycopg
        import uuid
        self.schema = "test_" + uuid.uuid4().hex
        url = os.environ["L2S1_TEST_POSTGRES_URL"]
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA "{self.schema}"')
        self.url = url
        self.reopen()

    def reopen(self):
        self.store = Store(self.url, options=f"-c search_path={self.schema}")

    def tearDown(self):
        self.store.run(f'DROP SCHEMA "{self.schema}" CASCADE')
        self.store.conn.commit()
        self.store.close()


if __name__ == "__main__":
    unittest.main()
