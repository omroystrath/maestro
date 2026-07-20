# VERTEX 2.0 (vendored)

This directory contains the VERTEX 2.0 brain-tissue simulator, vendored into MAESTRO as one of our
simulation engines. **It is third-party software under a non-commercial licence** (see
`license.txt` and the repo-root `NOTICE`). It is not covered by MAESTRO's own LICENSE.

- Upstream README: `README-UPSTREAM.md`
- DeepBrain adapter + driver: `deepbrain/` (this is the only place we add code)
- Change log for our additions: `deepbrain/CHANGES-DEEPBRAIN.md`

MAESTRO talks to VERTEX through `src/maestro/simulators/vertex_bridge.py`, which calls
`deepbrain/run_vertex_export.m`. If MATLAB is not installed, the bridge reports unavailable and the
CLI falls back to the analytic simulator, so the rest of the pipeline still runs.

Reference: Thornton, Hutchings & Kaiser, *Wellcome Open Research* 2019, VERTEX 2.0.
