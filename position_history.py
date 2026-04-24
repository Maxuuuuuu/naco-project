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
        if pos is "F":
            self.F_count += 1
        elif pos is "B":
            self.B_count += 1
        elif pos is "L":
            self.L_count += 1
        elif pos is "C":
            self.C_count += 1
        else:
            raise ValueError("Invalid position")