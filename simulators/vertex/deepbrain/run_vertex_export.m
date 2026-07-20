function run_vertex_export(workdir)
% RUN_VERTEX_EXPORT  DeepBrain driver that runs a VERTEX stimulation simulation
% and exports MAESTRO-shaped records.
%
%   run_vertex_export(WORKDIR)
%
% Reads   : WORKDIR/spec.json  (written by maestro.simulators.vertex_bridge)
% Writes  : WORKDIR/records_00001.mat, records_00002.mat, ...
%           each containing a struct `record` with fields the Python bridge expects:
%             record.subject_id     (char)
%             record.connectome     (RxR double)     % region-by-region coupling
%             record.region_labels  (cellstr, 1xR)
%             record.baseline_lfp    (optional, CxT)
%             record.amplitude_mV   (double)
%             record.duration_ms    (double)
%             record.on_times_ms    (1xK double)
%             record.off_times_ms   (1xK double)
%             record.protocol       (char)
%             record.response       (Rx1 double)     % per-region response to stimulation
%             record.markers        (struct of scalar fields)
%
% This is intentionally a THIN adapter. The heavy lifting (building the rat neocortex model,
% the electric-field stimulation, STDP/STP) lives in the VERTEX code that ships in the parent
% directory and is described in the VERTEX 2.0 paper. Wire the relevant setup + run scripts
% (e.g. ratSomatosensoryCortex/singlePulse.m) into the marked section below, then reduce them to
% the record fields above.
%
% For development without the full model, this driver can emit a lightweight placeholder so the
% Python side has something to load; set spec.params.placeholder = true to use it.

    if nargin < 1 || isempty(workdir)
        error('run_vertex_export:args', 'workdir is required');
    end

    spec = jsondecode(fileread(fullfile(workdir, 'spec.json')));
    nSamples = getfield_default(spec, 'n_samples', 1);
    params   = getfield_default(spec, 'params', struct());
    seed     = getfield_default(spec, 'seed', 0);
    rng(double(seed));

    placeholder = getfield_default(params, 'placeholder', false);

    for i = 1:nSamples
        if placeholder
            record = make_placeholder_record(params);
        else
            % ================= WIRE VERTEX HERE =================
            % 1. Build the model (see ratSomatosensoryCortex/ and setup scripts).
            % 2. Set the stimulation field + on/off times from `params`
            %      (amplitude_mV, pulse_width_ms, stim_onset_ms, protocol, density_scale).
            % 3. Run the simulation (runRatSimulation*.m).
            % 4. Reduce the LFP / recruitment / synaptic outputs into the `record` fields above.
            % Until wired, fall back to the placeholder so the pipeline stays runnable:
            record = make_placeholder_record(params);
            % ===================================================
        end

        outfile = fullfile(workdir, sprintf('records_%05d.mat', i));
        save(outfile, 'record', '-v7');
    end
end

% ---------------------------------------------------------------------------

function record = make_placeholder_record(params)
% Minimal, clearly-fake record so the Python bridge can be developed before the full
% VERTEX model is wired in. NOT a scientific output.
    R = getfield_default(params, 'n_regions', 29);
    amp = getfield_default(params, 'amplitude_mV', 750);
    W = abs(randn(R, R)); W = 0.5 * (W + W');  % symmetric pseudo-connectome
    W(1:R+1:end) = 0;
    drive = zeros(R, 1); drive(randi(R)) = amp / 750;
    resp = tanh(W * drive) + 0.05 * randn(R, 1);

    labels = cell(1, R);
    for k = 1:R, labels{k} = sprintf('R%d', k-1); end

    record = struct();
    record.subject_id    = sprintf('vertex-placeholder-%d', randi(1e6));
    record.connectome    = W;
    record.region_labels = labels;
    record.amplitude_mV  = amp;
    record.duration_ms   = getfield_default(params, 'pulse_width_ms', 0.5);
    record.on_times_ms   = getfield_default(params, 'stim_onset_ms', 1500);
    record.off_times_ms  = record.on_times_ms + record.duration_ms;
    record.protocol      = getfield_default(params, 'protocol', 'single_pulse');
    record.response      = resp;
    m = struct(); m.peak = max(resp); m.l2 = norm(resp);
    record.markers       = m;
end

function v = getfield_default(s, name, default)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end
