"""Two-window trip and three-window normal recovery state machine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtectionEvent:
    event: str
    state_id: int
    state_name: str
    confidence: float
    relay_cutoff: bool
    buzzer: bool
    red_led: bool


class FaultProtection:
    """Latch a protective trip after matching confident fault predictions."""

    def __init__(
        self,
        threshold: float = 0.85,
        trip_windows: int = 2,
        recovery_windows: int = 3,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if trip_windows < 1 or recovery_windows < 1:
            raise ValueError("window counts must be positive")
        self.threshold = threshold
        self.trip_windows = trip_windows
        self.recovery_windows = recovery_windows
        self.active = False
        self._candidate_state = None
        self._fault_count = 0
        self._normal_count = 0

    def update(self, state_id: int, confidence: float, state_name: str) -> ProtectionEvent | None:
        if not 0 <= state_id <= 4 or not 0.0 <= confidence <= 1.0:
            raise ValueError("state_id must be 0..4 and confidence must be in [0, 1]")
        if self.active:
            if state_id == 0 and confidence > self.threshold:
                self._normal_count += 1
            else:
                self._normal_count = 0
            if self._normal_count >= self.recovery_windows:
                self.active = False
                self._reset_counts()
                return self._event("recovery", state_id, state_name, confidence)
            return None

        if state_id != 0 and confidence > self.threshold:
            if state_id == self._candidate_state:
                self._fault_count += 1
            else:
                self._candidate_state = state_id
                self._fault_count = 1
        else:
            self._candidate_state = None
            self._fault_count = 0

        if self._fault_count >= self.trip_windows:
            self.active = True
            self._normal_count = 0
            return self._event("trip", state_id, state_name, confidence)
        return None

    def _reset_counts(self) -> None:
        self._candidate_state = None
        self._fault_count = 0
        self._normal_count = 0

    def _event(self, event: str, state_id: int, state_name: str, confidence: float) -> ProtectionEvent:
        return ProtectionEvent(
            event,
            state_id,
            state_name,
            confidence,
            self.active,
            self.active,
            self.active,
        )
