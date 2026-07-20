# DeepBrain changes to VERTEX

VERTEX is vendored under its own non-commercial licence (see ../license.txt and the repo NOTICE).
To keep the boundary clean, all DeepBrain additions live in this `deepbrain/` folder. We do not
edit upstream VERTEX files. Record every change here.

## Additions

- `run_vertex_export.m` — thin driver invoked by `maestro.simulators.vertex_bridge`. Reads a
  `spec.json`, runs a VERTEX stimulation simulation, and exports MAESTRO-shaped `record` structs to
  `records_*.mat`. Includes a placeholder mode so the Python pipeline is runnable before the full
  rat-neocortex model is wired in.

## TODO to make it "real"

- Wire the rat somatosensory cortex model (`../ratSomatosensoryCortex/`) and the single/paired/TBS
  stimulation scripts into `run_vertex_export.m` (see the marked section).
- Decide the canonical `response` reduction (per-region activity delta vs LFP decomposition).
- Add `density_scale` handling so desktop runs are tractable (paper suggests reducing density 10x).
