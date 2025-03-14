from enum import Enum
from numpy import double

class Grid_Size(Enum):
    BEGINNER = (9, 9)
    INTERMEDIATE = (16, 16)
    EXPERT = (30, 16)
    CUSTOM_LIMIT = (30, 24)
    RANKED_LIMIT = (50, 50)
    
class Mine_Density:
    BEGINNER = double(0.123)
    INTERMEDIATE = double(0.156)
    EXPERT = double(0.206)
    UPPER_LIMIT = double(668 / 720)