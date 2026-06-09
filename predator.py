from dataclasses import dataclass

@dataclass
class Predator:
    L_predation: float
    F_predation: float
    B_predation: float
    C_predation: float

    def normalize(self):
        total = self.L_predation + self.F_predation + self.B_predation + self.C_predation
        self.L_predation /= total
        self.F_predation /= total
        self.B_predation /= total
        self.C_predation /= total

        return self