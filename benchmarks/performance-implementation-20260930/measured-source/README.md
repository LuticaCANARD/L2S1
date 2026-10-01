# Historical measured source

These five files are the exact inputs hashed in `../implementation-hashes.json`,
from commit `e193b4c8ad32514318e917b90efaed2624051d71`. They belong to the completed
2026-09-30 measurement, not to later implementations. `../verify.py` checks their
hashes and recomputes the historical metrics with the measured metric code.
Current implementations are covered by the separate Rust/Python contract checks.
Do not update these snapshots or hashes to make a new implementation pass; record
new measurements separately.
