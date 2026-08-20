from __future__ import annotations

import enum

class IntegratedVerdictAction(enum.Enum):
    IGNORED = "ignored"
    NONE = "none"
    STARTED_PREDICTED = "started_predicted"
    COMMIT = "commit"
