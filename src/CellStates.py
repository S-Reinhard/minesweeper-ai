from enum import IntEnum
import numpy as np

class CELL_STATE(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    FORBIDDEN = 9
    COVERED = 10
    FLAGGED = 11
    MINE = 12
    
KNOWN_STATES = (
    CELL_STATE.ZERO,
    CELL_STATE.ONE,
    CELL_STATE.TWO,
    CELL_STATE.THREE,
    CELL_STATE.FOUR,
    CELL_STATE.FIVE,
    CELL_STATE.SIX,
    CELL_STATE.SEVEN,
    CELL_STATE.EIGHT,
    CELL_STATE.MINE
)

UNKNWON_STATES = (
    CELL_STATE.COVERED,
    CELL_STATE.FLAGGED
)

NUMBERED_STATES = (
    CELL_STATE.ZERO,
    CELL_STATE.ONE,
    CELL_STATE.TWO,
    CELL_STATE.THREE,
    CELL_STATE.FOUR,
    CELL_STATE.FIVE,
    CELL_STATE.SIX,
    CELL_STATE.SEVEN,
    CELL_STATE.EIGHT
)