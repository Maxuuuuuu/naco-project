# Define the possible positions for the agent
from enum import Enum


class Position(str, Enum):
    LEADER = "L"
    FOLLOWER = "F"
    BORDER = "B"
    CENTER = "C"

