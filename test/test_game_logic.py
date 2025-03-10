import pytest
from src.MinesweeperLogic import MinesweeperLogic
from src.GameSettings import GridSize, GridSize
import re

    
def test_init_game_with_default_settings():
    ml = MinesweeperLogic()
    
    # ensure that all required values are initialized
    assert ml.COLS is not None
    assert ml.ROWS is not None
    assert ml.SEED is not None
    assert ml.MINE_FIELDS is not None
    
    
def test_map_size_validation():
    cols, rows = GridSize.BOUNDARIES.value
    
    # test cases that should not be possible
    with pytest.raises(ValueError, match=re.escape("The number of columns (COLS) must be <= ")):
        MinesweeperLogic(COLS=cols+1)
    
    with pytest.raises(ValueError, match=re.escape("The number of rows (ROWS) must be <= ")):
        MinesweeperLogic(ROWS=rows+1)
    
    with pytest.raises(ValueError, match=re.escape("COLS must be > 0")):
        MinesweeperLogic(COLS=0)
    
    with pytest.raises(ValueError, match=re.escape("ROWS must be > 0")):
        MinesweeperLogic(ROWS=0)
    
    with pytest.raises(ValueError, match=re.escape("COLS must be > 0")):
        MinesweeperLogic(COLS=-1)
    
    with pytest.raises(ValueError, match=re.escape("ROWS must be > 0")):
        MinesweeperLogic(ROWS=-1)
    
    # test cases that are allowed
    MinesweeperLogic(COLS=1)
    MinesweeperLogic(ROWS=1)
    MinesweeperLogic(COLS=1, ROWS=1)
    MinesweeperLogic(COLS=GridSize.EXPERT.value[0])
    MinesweeperLogic(ROWS=GridSize.EXPERT.value[1])
    MinesweeperLogic(COLS=GridSize.EXPERT.value[0], ROWS=GridSize.EXPERT.value[1])
        
        

def test_detailed_game_init():
    pass
    
    
        