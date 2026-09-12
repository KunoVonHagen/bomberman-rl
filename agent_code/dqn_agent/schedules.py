from __future__ import annotations


class LinearSchedule:
    """Linearly interpolate from start to end over a fixed duration."""

    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, int(duration))

    def value(self, step: int) -> float:
        """Return the schedule value at the current step."""
        fraction = min(1.0, max(0.0, step / self.duration))
        return self.start + fraction * (self.end - self.start)