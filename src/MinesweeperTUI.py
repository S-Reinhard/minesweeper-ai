from src.MinesweeperLogic import MinesweeperLogic


class MinesweeperTUI:
    
    def __init__(self, logic:MinesweeperLogic):
        self.logic = logic
        
    
    def __str__(self):
        
        lines = []
        logic = self.logic
        
        for r in range(logic.MAP_BOUNDARIES_ROWS):
            line = self.logic.GRID[r * logic.MAP_BOUNDARIES_COLS: (r+1)*logic.MAP_BOUNDARIES_COLS]