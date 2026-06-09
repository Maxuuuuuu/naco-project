# Store position counts.
from position import Position


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

    def reset(self):
        self.L_count = 0
        self.F_count = 0
        self.B_count = 0
        self.C_count = 0

    def total(self):
        return self.F_count + self.B_count + self.L_count + self.C_count