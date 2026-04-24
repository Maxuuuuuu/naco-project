from dataclasses import dataclass

from prototype_dev import Position


# Store position counts.
@dataclass
class PositionHistory:
    F_count: int = 0
    B_count: int = 0
    L_count: int = 0
    C_count: int = 0

    def add(self, pos: Position):
        if pos == Position.FOLLOWER:
            self.F_count += 1
        elif pos == Position.BORDER:
            self.B_count += 1
        elif pos == Position.LEADER:
            self.L_count += 1
        elif pos == Position.CENTER:
            self.C_count += 1
        else:
            raise ValueError("Invalid position")