from enum import IntEnum

class CELL_STATE(IntEnum):
    ZERO    = 1 << 0
    ONE     = 1 << 1
    TWO     = 1 << 2
    THREE   = 1 << 3
    FOUR    = 1 << 4
    FIVE    = 1 << 5
    SIX     = 1 << 6
    SEVEN   = 1 << 7
    EIGHT   = 1 << 8
    FLAGGED = 1 << 9
    COVERED = 1 << 10
    FORBIDDEN = 1 << 11

    UNKNOWN = FLAGGED | COVERED
    KNOWN = ZERO | ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT
    NUMBERED = ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT
    ALLOWED = ZERO | ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT | FLAGGED | COVERED
    
    def __contains__(self, item):
        if not isinstance(item, CELL_STATE):
            return False
        return (self & item) == item

