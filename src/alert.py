from __future__ import annotations


class AlertPlayer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def play(self) -> None:
        if not self.enabled:
            return

        try:
            import winsound

            winsound.Beep(1200, 220)
        except Exception:
            print("\a", end="")
