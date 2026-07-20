import numpy as np
import pytest

from maestro.data.schema import Brain, Modality, Outcome, Record, Source, Stimulation


def test_record_validates_square_connectome():
    r = Record(
        brain=Brain(
            subject_id="s", connectome=np.eye(4), region_labels=[f"R{i}" for i in range(4)]
        ),
        stimulation=Stimulation(modality=Modality.EFIELD),
        outcome=Outcome(response=np.zeros(4)),
        source=Source.SIMULATED,
    )
    r.validate()  # should not raise
    assert r.is_labelled()


def test_record_rejects_mismatched_response():
    r = Record(
        brain=Brain(subject_id="s", connectome=np.eye(4)),
        stimulation=Stimulation(modality=Modality.EFIELD),
        outcome=Outcome(response=np.zeros(5)),  # wrong size
        source=Source.SIMULATED,
    )
    with pytest.raises(ValueError):
        r.validate()
