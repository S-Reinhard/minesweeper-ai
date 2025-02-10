import gymnasium as gym
import numpy as np
from numpy import arange as range
from gymnasium.spaces import MultiDiscrete, Dict, Box, Discrete
from enum import IntEnum
from typing import Optional

class Action():
    REVEAL = 0
    FLAG = 1
    UNFLAG = 2
    actions = [REVEAL, FLAG, UNFLAG]

class State():
    FORBIDDEN = -3
    FLAGGED = -2
    UNKNOWN = -1
    EMPTY = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    
class Mine_Exception(Exception):
    def __init__(self, message, field, col, row):
        super().__init__(message)
        
        self.field = field
        self.col = col
        self.row = row

class MinesweeperEnv(gym.Env):

    def __init__(self, size=(9,9), mine_ratio=0.15625, seed=None):
        super().__init__()

        # init random
        if (seed == None):
            seed = self._generate_seed()
        self.seed = seed
        

        # Set Grid data
        self.COLS = self._rand_col_num(size) + 2
        self.ROWS = self._rand_row_num(size) + 2
        field_num = self.COLS*self.ROWS
        self.FIELD_NUM = field_num
        self.FIELDS = range(0, field_num)    

        # Set Meta Spaces
        self.observation_space = Dict({
            "Forbidden": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "Allowed": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "Flagged": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "Unknown": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "Empty": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "ONE": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "TWO": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "THREE": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "FOUR": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "FIVE": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "SIX": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "SEVEN": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "EIGHT": Box(low=0, high=1, shape=(field_num,), dtype=np.int8),
            "GRID": Box(low=-3, high=8, shape=(field_num,), dtype=np.int8),
            "Revealed_Percent": Box(low=0.0, high=100.0, shape=(1,), dtype=np.float256),
        })
        self.action_space = MultiDiscrete([len(Action.actions), self.COLS, self.ROWS])

        

    def _generate_map(self):
        # init map
        self.grid = np.zeros(shape=(self.COLS*self.ROWS,))
        self.FORBIDDEN_FIELDS:np.ndarray = self.get_forbidden_fields(self.COLS, self.ROWS)
        self.ALLOWED_FIELDS:np.ndarray = self.get_allowed_fields(self.COLS, self.ROWS)

        self.grid[self.ALLOWED_FIELDS] = State.UNKNOWN
        self.grid[self.FORBIDDEN_FIELDS] = State.FORBIDDEN


        # Distribute mines
        rng = self._get_rng()
        mine_num = len(self.ALLOWED_FIELDS) * self.get_mine_ratio()
        self.MINE_FIELDS = rng.choice(self.ALLOWED_FIELDS, mine_num)
        
        # calc number map
        self.bomb_counts = {}
        for one_field in self.FIELDS:
            self.bomb_counts[one_field] = self.get_bomb_count_of_field(one_field)



    def get_forbidden_fields(self, cols, rows) -> np.ndarray:
        forbidden_fields = []
        forbidden_fields += range(0, cols)
        forbidden_fields += range(cols * (rows - 1), cols * rows)
        forbidden_fields += range(0,cols * (rows - 1), cols)
        forbidden_fields += range(cols -1, cols * rows, cols)
        return np.unique(forbidden_fields)
    
    def get_allowed_fields(self, cols, rows):
        return np.setdiff1d(range(0, cols * rows), self.get_forbidden_fields(cols, rows))

    def get_allowed_grid(self)->np.ndarray:
        return self.grid[self.ALLOWED_FIELDS]


    def count_revealed_fields(self):
        np.array(self.get_allowed_grid() >= State.EMPTY, dtype=bool).sum()


    def _generate_seed(self):
        return np.random.randint(0, 2**32 - 1)
    
   
    def _get_rng(self):
        return np.random.default_rng(self.seed)
    

    def get_mine_ratio(self, mine_ratio=None):
        rng = self._get_rng()
        if(mine_ratio is None or mine_ratio < 0 or mine_ratio > 1):
            mine_ratio = rng.uniform(low=0.12345679012345678, high=0.20625)
        return mine_ratio

    def _rand_row_num(self, size):
        rng = self._get_rng()
        return size[1] if (size[1] is not None and size[0] > 0) else rng.integers(low=8, high=16, endpoint=True)

    def _rand_col_num(self, size):
        rng = self._get_rng()
        return size[0] if (size[0] is not None and size[0] > 0) else rng.integers(low=8, high=30, endpoint=True)

    def get_neighbor_fields(self, field:int):
        if (field in self.ALLOWED_FIELDS):
            return (
                field + 1,
                field + self.COLS + 1,
                field + self.COLS,
                field + self.COLS - 1,
                field - 1,
                field - self.COLS - 1,
                field - self.COLS,
                field - self.COLS + 1
            )
        else:
            return None
        
    def get_bomb_count_of_field(self, field: int) -> np.uint8:
        bomb_count: np.uint8 = 0
        
        # check if field is forbidden
        if field in self.FORBIDDEN_FIELDS:
            return -1
        
        # get list of neighboring fields
        neigbors = self.get_neighbor_fields(field)
        
        # count bombs on neiboring fields
        for one_field in neigbors:
            if one_field in self.MINE_FIELDS:
                bomb_count += 1
        
        # return bombs
        return bomb_count
    
    def get_coords_of_field(self, field: int):
        return (field % self.COLS, int(field/self.COLS))

    def reveal(self, field:int):
        # add list to fields to the - to be revealed queue

        # check if field is a mine
        if field in self.MINE_FIELDS:
            col, row = self.get_coords_of_field(field)
            raise Mine_Exception("Stepped on Mine", field, col, row)
        
        to_be_revealed = []
        to_be_revealed.append(field)
        
        while len(to_be_revealed) > 0:
            field = to_be_revealed.pop(0)

            # check if field can be revealed
            if self.grid[field] == State.UNKNOWN:
                bomb_count = self.bomb_counts[field]

                # reveal bomb count
                self.grid[field] = bomb_count

                # reveal neigbors if bomb count == 0 
                if bomb_count == 0:
                    to_be_revealed.append(self.get_neighbor_fields(field))


    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # generate new seed if no seed is passed
        if seed is None:
            self.seed = self._generate_seed()
        else:
            self.seed = seed

        # create a new map
        self._generate_map()

        # return mandatory data
        return self._get_ops(), self._get_info()


    def step(self, action):
        # TODO
        return super().step(action)

    def render(self):
        # TODO
        return super().render()

    def close(self):
        # TODO
        return super().close()
    
    def _get_ops(self):
        return {
            "Forbidden": np.array(self.grid == State.FORBIDDEN, dtype=np.int8),
            "Allowed": np.array(self.grid != State.FORBIDDEN, dtype=np.int8),
            "Flagged": np.array(self.grid==State.FLAGGED, dtype=np.int8),
            "Unknown": np.array(self.grid==State.UNKNOWN, dtype=np.int8),
            "Empty": np.array(self.grid==State.EMPTY, dtype=np.int8),
            "ONE": np.array(self.grid==State.ONE, dtype=np.int8),
            "TWO": np.array(self.grid==State.TWO, dtype=np.int8),
            "THREE": np.array(self.grid==State.THREE, dtype=np.int8),
            "FOUR": np.array(self.grid==State.FOUR, dtype=np.int8),
            "FIVE": np.array(self.grid==State.FIVE, dtype=np.int8),
            "SIX": np.array(self.grid==State.SIX, dtype=np.int8),
            "SEVEN": np.array(self.grid==State.SEVEN, dtype=np.int8),
            "EIGHT": np.array(self.grid==State.EIGHT, dtype=np.int8),
            "GRID": np.array(self.grid, dtype=np.int8),
            "Revealed_Percent": [self.count_revealed_fields()/len(self.get_allowed_grid())],
        }
    
    def _get_info(self):
        # TODO
        return
    
         
    

