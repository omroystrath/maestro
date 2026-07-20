# Data governance

MAESTRO will eventually train on real, consented human and animal data. The rules below apply from
day one so good habits are in place before any sensitive data arrives.

## Hard rules

1. **Nothing sensitive in git.** No real subject data, no identifiers, no raw clinical records.
   The `data/` folders are gitignored. Only `.gitkeep` placeholders are tracked.
2. **Every record carries a `source`** (`simulated`, `synthetic`, `real`). Real data is handled
   under its own access controls and never mixed into public artefacts.
3. **Share the statistics, never the patient.** The flow-matching generator lets us produce and
   share `synthetic` cohorts that carry population statistics without identities. Prefer sharing
   synthetic data over moving real records.
4. **Consent scope is tracked with the data**, not in code. The loader for real data must refuse
   records whose consent scope does not cover the current use.

## Where data lives

| Folder | Contents | Tracked? |
| ------ | -------- | -------- |
| `data/raw/` | immutable inputs (public connectome/imaging, exported sim results) | no |
| `data/interim/` | Records produced by simulators | no |
| `data/processed/` | model-ready tensors/splits | no |
| `data/synthetic/` | generator-sampled Records, shareable | no (curated subsets shared out-of-band) |

## Real data (when we get there)

- Stored in an access-controlled store, not in the repo.
- Loaders live behind an interface that checks consent scope and logs access.
- De-identification and governance sign-off happen before data enters `data/`.

If in doubt, treat it as sensitive and ask before committing.
