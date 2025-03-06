from src.MinesweeperLogic import MinesweeperLogic
from src.GameSettings import Grid_Size

class MinesweeperGame():
    
    def __init__(self):
        # create new game
        current_state = MinesweeperLogic(COLS=Grid_Size.BEGINNER[0], ROWS=Grid_Size.BEGINNER[1])
        