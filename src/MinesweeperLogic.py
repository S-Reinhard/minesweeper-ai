from __future__ import annotations
import numpy as np
from numpy import uint8, uint16, ndarray
from numpy.random import SeedSequence, BitGenerator, Generator
from numpy.typing import NDArray
from collections.abc import Iterable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
from src.CellStates import CELL_STATE, KNOWN_STATES, NUMBERED_STATES
from src.GameSettings import GridSize, MineDensity, GridSize
import secrets

class Res_Code(Enum):
    Successful = auto()
    Forbidden = auto()
    REVEALED_AND_STEPPED_ON_MINE = auto()
    CHORED_AND_STEPPED_ON_MINE = auto()
    TRIED_TO_REVEAL_FORBIDDEN_CELL = auto()
    TRIED_TO_REVEAL_FLAGGED_CELL = auto()
    TRIED_TO_REVEAL_KNOWN_CELL = auto()
    REVEALING_WAS_SUCCESSFUL = auto()
    TRIED_TO_CHORD_FORBIDDEN_CELL = auto()
    TRIED_TO_CHORD_UNNUMBERED_CELL = auto()
    TRIED_TO_CHORD_EMPTY_CELL = auto()
    TRIED_TO_CHORD_UNSATURATED_CELL = auto()
    TRIED_TO_CHORD_REVEALED_CELLS = auto()
    CHOR_WAS_SUCCESSFUL = auto()
    TRIED_TO_FLAG_FORBIDDEN_CELL = auto()
    TRIED_TO_FLAG_REAVEALED_CELL = auto()
    TRIED_TO_FLAG_FLAGGED_CELL = auto()
    TRIED_TO_FLAG_UNCOVERED_CELL = auto()
    FLAG_SUCESSFULLY_PLACED = auto()
    TRIED_TO_UNFLAG_FORBIDDEN_CELL = auto()
    TRIED_TO_UNFLAG_REAVEALED_CELL = auto()
    TRIED_TO_UNFLAG_COVERED_CELL = auto()
    TRIED_TO_UNFLAG_UNFLAGGED_CELL = auto()
    FLAG_SUCESSFULLY_REMOVED = auto()
    
    
class AutoRevealedMine(Exception):
    """Exception raised for custom error scenarios.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

@dataclass(frozen=True)
class MinesweeperLogic:
    """
    A class for initializing and managing a Minesweeper game board.

    This class encapsulates the logic to generate a Minesweeper board with randomized or pre-defined dimensions,
    mine placements, and an optional seed for random number generation. It performs several validations to ensure
    that the game board is generated within acceptable constraints, including map boundary sizes, proper mine placement,
    and valid data types for mines.

    Attributes:
        COLS (uint8): The number of columns for the active game board. If None, a random value within the specified
                      range [LOWER_COL_RANGE, UPPER_COL_RANGE] is generated.
        ROWS (uint8): The number of rows for the active game board. If None, a random value within the specified
                      range [LOWER_ROW_RANGE, UPPER_ROW_RANGE] is generated.
        MINE_FIELDS (ndarray): An array specifying the positions of mines on the board. If None, mine locations are
                               determined randomly based on a generated mine density. If provided, the data type is
                               validated to be np.uint16 and the mine positions are checked against forbidden fields.
        SEED (None | int | List[int] | SeedSequence | BitGenerator | Generator): A seed or random state for the game.
                               If not provided, a random 64-bit seed is generated using a secure method.
        LOWER_COL_RANGE (uint8): The lower boundary for the randomly generated number of columns (default is based on
                                 Grid_Size.BEGINNER.value[0]).
        UPPER_COL_RANGE (uint8): The upper boundary for the randomly generated number of columns (default is based on
                                 Grid_Size.EXPERT.value[0]).
        LOWER_ROW_RANGE (uint8): The lower boundary for the randomly generated number of rows (default is based on
                                 Grid_Size.BEGINNER.value[1]).
        UPPER_ROW_RANGE (uint8): The upper boundary for the randomly generated number of rows (default is based on
                                 Grid_Size.EXPERT.value[1]).
        MAP_BOUNDARIES_COLS (uint8): The absolute column boundary for the overall map, which must be at least
                                     UPPER_COL_RANGE + 2.
        MAP_BOUNDARIES_ROWS (uint8): The absolute row boundary for the overall map, which must be at least
                                     UPPER_ROW_RANGE + 2.

    Methods:
        __post_init__():
            Performs post-initialization processing that includes:
                1. Generating a random seed if none is provided.
                2. Validating that the map boundaries are large enough to accommodate the generated board perimeters.
                3. Checking that mine fields are not defined before the board dimensions (COLS and ROWS) are set.
                4. Randomly generating COLS and ROWS if they are not already defined.
                5. Validating that the overall map size is sufficiently larger than the active board dimensions.
                6. Randomly placing mines if MINE_FIELDS is not provided, using a randomly selected mine density.
                7. If MINE_FIELDS is provided, validating its data type and ensuring all mines are on allowed fields.

    Raises:
        ValueError: If any of the following conditions occur:
                    - The map boundaries are too small to cover the generated level perimeters.
                    - Mine fields are provided before defining the board dimensions.
                    - The overall map size is inadequate relative to the active board dimensions.
                    - Mines are placed on forbidden cells.
        TypeError: If MINE_FIELDS does not have the required numpy data type (np.uint16).
    """
    COLS: uint8 = field(default=None)
    ROWS: uint8 = field(default=None)
    MINE_FIELDS: ndarray = field(default=None)
    SEED: None | int | list[int] | SeedSequence | BitGenerator | Generator = field(default=None)
    
    # size boudaries for the generated levels. 
    LOWER_COL_RANGE: uint8 = field(default=GridSize.BEGINNER.value[0])
    UPPER_COL_RANGE: uint8 = field(default=GridSize.EXPERT.value[0])
    LOWER_ROW_RANGE: uint8 = field(default=GridSize.BEGINNER.value[1])
    UPPER_ROW_RANGE: uint8 = field(default=GridSize.EXPERT.value[1])
    
    # absolute size boundaries for the map.
    MAP_BOUNDARIES_COLS: uint8 = field(default=GridSize.TRUE_COLS())
    MAP_BOUNDARIES_ROWS: uint8 = field(default=GridSize.TRUE_ROWS())
    
    GRID: NDArray[np.int8] = field(default=None) # hack!!! Grid contains Field_State not integers, but Field_States is in IntEnum and theirfor the values behave like integers.  
    
    def __post_init__(self):
        # if no seed was given generate one
        if (self.SEED is None):
            object.__setattr__(self, 'SEED', secrets.randbits(64))
            
        # Validate if MAP_BOUNDARIES are big enough to cover the perimeters of the generated levels
        self._check_map_generation_boundaries()

        # Validate that mine fields are not provided when the board dimensions (COLS and ROWS) are missing.
        self._check_too_early_mine_definition()
            
        # init Size if None
        if (self.COLS is None):
            random_col_num = self.RNG.integers(self.LOWER_COL_RANGE, self.UPPER_COL_RANGE, endpoint=True)
            object.__setattr__(self, "COLS", random_col_num) 
        if (self.ROWS is None):
            random_row_num = self.RNG.integers(self.LOWER_ROW_RANGE, self.UPPER_ROW_RANGE, endpoint=True)
            object.__setattr__(self, "ROWS", random_row_num)

        # validate Map Size
        self._check_too_small_map_size()
        
        # if no mine fields are given place mines randomly
        if (self.MINE_FIELDS is None):
            # select random mine ration
            mine_ratio = self.RNG.uniform(MineDensity.BEGINNER, MineDensity.EXPERT)
            mine_num = uint16(self.SIZE * mine_ratio)
            mine_fields = self.RNG.choice(self.ALLOWED_CELLS, mine_num, replace=False)
            object.__setattr__(self, 'MINE_FIELDS', mine_fields)
            self.MINE_FIELDS.setflags(write=False)
            
        else:
            # Validate the Type of the mine array
            self._check_mine_fields_dtype()
                
            # Validate that all mines are on allowed fields
            self._check_forbidden_mines()
        
        
        # if grid is none -> calculate grid; else validate 
        if (self.GRID is None):
             # init grid as forbidden 
             grid: ndarray[CELL_STATE] = np.full(self.TRUE_SIZE, CELL_STATE.FORBIDDEN, dtype=CELL_STATE)
             object.__setattr__(self, 'GRID', grid)
             
             # set allowed cells to coverd
             self.GRID[self.ALLOWED_CELLS] = CELL_STATE.COVERED
        else:
            self._validate_grid_size()
            
        self.GRID.setflags(write=False)

    def _validate_grid_size(self):
        if (len(self.GRID) != self.TRUE_SIZE):
            raise ValueError(
                    f"Calculated Grid Size mismatch: "
                    f"Expected a grid size of {self.TRUE_SIZE} but got {len(self.GRID)}. "
                    f"This issue may be caused by a discrepancy in the board dimensions (COLS and ROWS) or an incorrect grid initialization. "
                    f"Please ensure the grid size matches the defined dimensions."
                )
        
    def _check_map_generation_boundaries(self):
        """
        Validates that the map boundaries are sufficiently larger than the upper range limits for columns and rows.

        The function computes the extra space available in both columns and rows by subtracting the upper range
        from the defined map boundaries. If the extra space in either dimension is less than 2, it raises a ValueError,
        indicating that the boundaries are too tight to properly cover the generated level perimeters.

        Raises:
            ValueError: If either the column or row boundary extra space is less than 2. The error message
                        specifies which boundary is too small and suggests the minimum required size.
        """
        # Calculate extra space available in columns and rows
        extra_space_col = self.MAP_BOUNDARIES_COLS - self.UPPER_COL_RANGE
        extra_space_row = self.MAP_BOUNDARIES_ROWS - self.UPPER_ROW_RANGE

        # Check if the extra space is insufficient in either dimension
        if (extra_space_col < 2) or (extra_space_row < 2):
            error_message = "MAP_BOUNDARIES are too small to cover the generated level perimeters.\n"

            # Check column boundary and append specific error message if too small
            if self.MAP_BOUNDARIES_COLS - self.UPPER_COL_RANGE < 2:
                error_message += (
                    f"- Column boundary is too small: MAP_BOUNDARIES_COLS={self.MAP_BOUNDARIES_COLS}, "
                    f"UPPER_COL_RANGE={self.UPPER_COL_RANGE}. Adjust MAP_BOUNDARIES_COLS to be at least {self.UPPER_COL_RANGE + 2}.\n"
                )

            # Check row boundary and append specific error message if too small
            if self.MAP_BOUNDARIES_ROWS - self.UPPER_ROW_RANGE < 2:
                error_message += (
                    f"- Row boundary is too small: MAP_BOUNDARIES_ROWS={self.MAP_BOUNDARIES_ROWS}, "
                    f"UPPER_ROW_RANGE={self.UPPER_ROW_RANGE}. Adjust MAP_BOUNDARIES_ROWS to be at least {self.UPPER_ROW_RANGE + 2}.\n"
                )

            raise ValueError(error_message)

    def _check_forbidden_mines(self):
        """
        Checks for any mine placements that fall on forbidden fields.

        Uses the isForbidden method to determine if any cell indices in MINE_FIELDS correspond to fields 
        where mines should not be placed. If any forbidden mine placements are detected, a ValueError is raised 
        with details of the offending cell indices.

        Raises:
            ValueError: If one or more mine fields are placed on forbidden cells.
        """
        # Determine if any mine field is in a forbidden area
        if (np.any(self._isForbidden(self.MINE_FIELDS))):
            forbidden_mines = self.MINE_FIELDS[self._isForbidden(self.MINE_FIELDS)]
            raise ValueError(
                f"Invalid MINE_FIELDS: the following cell indices are placed on forbidden fields: {forbidden_mines}. "
                "Please ensure that all mines are located on allowed fields according to the board dimensions and rules."
            )

    def _check_mine_fields_dtype(self):
        """
        Validates that the data type of MINE_FIELDS is np.uint16.

        The function checks whether the numpy data type of MINE_FIELDS matches the expected type. If not, it raises
        a TypeError with an informative message.

        Raises:
            TypeError: If MINE_FIELDS does not have the numpy data type 'np.uint16'.
        """
        # Check if the dtype of MINE_FIELDS is not np.uint16
        if (self.MINE_FIELDS.dtype != np.dtype(uint16)):
            raise TypeError(
                f"MINE_FIELDS must have dtype 'np.uint16', but got {self.MINE_FIELDS.dtype}."
            )

    def _check_too_small_map_size(self):
        """
        Ensures that the overall map size is sufficiently larger than the active game board dimensions.

        The function verifies that both the number of columns and rows in the map boundaries exceed the game board's 
        COLS and ROWS by at least 2. This is necessary to maintain an adequate buffer around the board.

        Raises:
            ValueError: If the map boundaries for either columns or rows are less than COLS + 2 or ROWS + 2, respectively.
        """
        # Verify that the map has at least a 1-cell border on each side
        if (self.MAP_BOUNDARIES_COLS - self.COLS < 2):
            raise ValueError(f"The number of columns (COLS) must be <= {self.MAP_BOUNDARIES_COLS-2} to ensure that there is at least one column of forbidden cells to the left and right of the playable area")
        if (self.MAP_BOUNDARIES_ROWS - self.ROWS < 2):
            raise ValueError(f"The number of rows (ROWS) must be <= {self.MAP_BOUNDARIES_ROWS-2} to ensure that there is at least one row of forbidden cells to the top and bottom of the playable area")
        
        if (self.ROWS <= 0):
            raise ValueError("ROWS must be > 0")             
        if (self.COLS <= 0):
            raise ValueError("COLS must be > 0")             
    
    def _check_too_early_mine_definition(self):
        """
        Validates that MINE_FIELDS is not defined before the essential board dimensions are set.

        This function checks that if MINE_FIELDS is provided (i.e., not None), then both COLS and ROWS must be defined.
        If either COLS or ROWS is None while MINE_FIELDS is provided, it raises a ValueError to prevent early or 
        improper mine definition.

        Raises:
            ValueError: If MINE_FIELDS is provided while COLS or ROWS is None.
        """
        # Check if MINE_FIELDS is set before board dimensions are established
        if ((self.COLS is None or self.ROWS is None) and self.MINE_FIELDS is not None):
            raise ValueError(
                "Invalid parameters: MINE_FIELDS cannot be provided when COLS or ROWS is None. "
                "Please provide both COLS and ROWS when specifying MINE_FIELDS."
            )

    def _by_state(self, cells: NDArray[np.int8], state):
        return

    @property
    def TRUE_SIZE(self) -> uint16:
        return self.MAP_BOUNDARIES_COLS * self.MAP_BOUNDARIES_ROWS
    
    @property
    def SIZE(self) -> uint16:
        return self.COLS * self.ROWS
    
    @property
    def CELLS(self) -> NDArray[uint16]:
        cells = np.arange(self.TRUE_SIZE, dtype=uint16)
        cells.setflags(write=False)
        return cells
        
    @property
    def ALLOWED_CELLS(self) -> NDArray[uint16]:
        allowed_mask = self._isAllowed(self.CELLS)
        allowed_fields: ndarray = self.CELLS[allowed_mask]
        allowed_fields.setflags(write=False)
        return allowed_fields
    
    @property
    def FORBIDDEN_CELLS(self) -> NDArray[uint16]:
        forbidden_cells: ndarray = self.CELLS[self._isForbidden(self.CELLS)]
        forbidden_cells.setflags(write=False)
        return forbidden_cells

    @property
    def MINE_NUM(self) -> uint16:
        return len(self.MINE_FIELDS)
    
    @property
    def RNG(self) -> np.random.Generator:
        return np.random.default_rng(seed=self.SEED)

    @property
    def CELL_NUMBERS(self) -> NDArray[uint16]:
        return self._getNumber(self.ALLOWED_CELLS)


    def _getNeighbors(self, cell: uint16):
        """
        Retrieve the neighboring cell indices for a given cell in the Minesweeper board.

        This function computes and returns the eight neighboring cell indices for the specified cell,
        using the board's column count (COLS) to determine relative positions. The neighbors are returned
        in clockwise order beginning on the top corresponding to their directional indices as follows:
          - 0: Top neighbor (cell - COLS)
          - 1: Top right neighbor (cell - COLS + 1)
          - 2: Right neighbor (cell + 1)
          - 3: Bottom right neighbor (cell + COLS + 1)
          - 4: Bottom neighbor (cell + COLS)
          - 5: Bottom left neighbor (cell + COLS - 1)
          - 6: Left neighbor (cell - 1)
          - 7: Top left neighbor (cell - COLS - 1)

        The function first checks whether the provided cell index is allowed using the isAllowed method.
        If the cell is not allowed, it returns None.

        Args:
            cell (uint16): The index of the cell for which neighboring cells are to be retrieved.

        Returns:
            ndarray of uint16 containing eight neighboring cell indices in the specified order if the cell is allowed;
                           otherwise, empty ndarray.
        """
        if(self._isAllowed(cell)):
            return np.array([
               cell - self.MAP_BOUNDARIES_COLS,
               cell - self.MAP_BOUNDARIES_COLS + 1,
               cell + 1,
               cell + self.MAP_BOUNDARIES_COLS + 1,
               cell + self.MAP_BOUNDARIES_COLS,
               cell + self.MAP_BOUNDARIES_COLS - 1,
               cell - 1,
               cell - self.MAP_BOUNDARIES_COLS - 1,
            ], dtype=uint16)
        else:
            return np.array([], dtype=uint16)

    def _isAllowed(self, cell: uint16) -> bool: 
        row_of_cell = cell // self.MAP_BOUNDARIES_COLS  
        col_of_cell = cell % self.MAP_BOUNDARIES_COLS

        return (0 < col_of_cell) & (col_of_cell <= self.COLS) & (0 < row_of_cell) & (row_of_cell <= self.ROWS)
    
    def _isForbidden(self, cell:uint16) -> bool:
        return self._isAllowed(cell) != True

    def _getNumber(self, cell:uint16) -> np.uint8:
        neighbors = self._getNeighbors(cell)
        mask = np.isin(neighbors, self.MINE_FIELDS)
        return np.sum(mask)
    
    def revealCell(self, cell: uint16) -> tuple[Res_Code, MinesweeperLogic]:
        # Check for various conditions that prevent revealing the cell.
        if np.isin(cell, self.MINE_FIELDS):
            return (Res_Code.REVEALED_AND_STEPPED_ON_MINE, self)
        if self._isForbidden(cell):
            return (Res_Code.TRIED_TO_REVEAL_FORBIDDEN_CELL, self)
        if self.GRID[cell] == CELL_STATE.FLAGGED:
            return (Res_Code.TRIED_TO_REVEAL_FLAGGED_CELL, self)
        if np.isin(self.GRID[cell], KNOWN_STATES):
            return (Res_Code.TRIED_TO_REVEAL_KNOWN_CELL, self)

        # create an updated grid where the cell is revealed
        NEW_GRID = self.reaveal_logic(cell)
        
        # return success code and updated game state
        return (Res_Code.REVEALING_WAS_SUCCESSFUL, self.update_grid(NEW_GRID))

    def reaveal_logic(self, init_cells: uint16 | NDArray[uint16]):
        NEW_GRID = self.GRID.copy()
        NEW_GRID.setflags(write=True)
        tmp_arr = np.atleast_1d(np.asarray(init_cells, dtype=uint16))
        cells_to_be_revealed = deque(tmp_arr)
        processed_cells = []

        # Process cells until there are no more cells to reveal
        while len(cells_to_be_revealed) > 0:
            one_cell = cells_to_be_revealed.popleft()
            
            if one_cell not in processed_cells and NEW_GRID[one_cell] == CELL_STATE.COVERED:
                # Raise an error if a mine is auto-revealed (this should never happen)
                if (one_cell in self.MINE_FIELDS):
                    raise AutoRevealedMine("This error indicates, that a mine got revealed, by auto reveal. This indicates, that either getNumber or the algorithm to reveal cells is bugged. And needs closer attention.")
                
                # Mark the cell as processed and update its state based on the number of adjacent mines
                processed_cells.append(one_cell)
                NUM = self._getNumber(one_cell)
                if 0 <= NUM <= 8:
                    NEW_GRID[one_cell] = CELL_STATE(NUM)
                    
                # If no adjacent mines, add neighboring allowed cells to the reveal queue
                if NUM == 0:
                    neighbors = np.asarray(self._getNeighbors(one_cell))
                    neighbors = self.get_allowed_of(neighbors)
                    cells_to_be_revealed.extend(neighbors)
        return NEW_GRID

    def get_allowed_of(self, cells:ndarray):
        return cells[self._isAllowed(cells)]
    
    def _getFlagNumber(self, cell:uint16) -> uint8:
            neighbors = self._getNeighbors(cell)
            neighbors = self.get_allowed_of(neighbors)
            arr = self.GRID[neighbors]
            return np.sum(arr == CELL_STATE.FLAGGED)
            
    
    def chord(self, cell:uint16) -> tuple[Res_Code, MinesweeperLogic]:
        
        # ensure that cell is not Forbidden
        if (self._isForbidden(cell)):
            return (Res_Code.TRIED_TO_CHORD_FORBIDDEN_CELL, self)
        
        # ensure that cell is numbered but not empty
        if (self.GRID[cell] in NUMBERED_STATES):
            return (Res_Code.TRIED_TO_CHORD_UNNUMBERED_CELL, self)
        
        # ensure that number of flags matches the number on the field
        if (self.CELL_NUMBERS[cell] != self._getFlagNumber(cell)):
            return (Res_Code.TRIED_TO_CHORD_UNSATURATED_CELL, self)
        
        # Get covered neighbors
        neighbors = self._getNeighbors(cell)
        neighbors = self.get_allowed_of(neighbors)
        covered_neighbors = self.get_cells_of_state(neighbors, CELL_STATE.COVERED)
        
        if len(covered_neighbors) == 0:
            return (Res_Code.TRIED_TO_CHORD_REVEALED_CELLS, self)
                
        # create grid where the cells are revealed
        NEW_GRID = self.reaveal_logic(covered_neighbors)
        
        # return success code and updated game state
        return (Res_Code.CHOR_WAS_SUCCESSFUL, self.update_grid(NEW_GRID))
    
    def place_flag(self, cell:uint16) -> tuple[Res_Code, MinesweeperLogic]:
        # check if field is Forbidden
        if (self._isForbidden(cell)):
            return (Res_Code.TRIED_TO_FLAG_FORBIDDEN_CELL, self)
        
        # check if field is already reveald
        if (self.GRID[cell] in KNOWN_STATES):
            return (Res_Code.TRIED_TO_FLAG_REAVEALED_CELL, self)
        
        # check if field is already flagged
        if (self.GRID[cell] == CELL_STATE.FLAGGED):
            return (Res_Code.TRIED_TO_FLAG_FLAGGED_CELL, self)
        
        # check if field is covered
        if (self.GRID[cell] != CELL_STATE.COVERED):
            return (Res_Code.TRIED_TO_FLAG_UNCOVERED_CELL, self)
        
        # place flag
        NEW_GRID = self.GRID.copy()
        NEW_GRID.setflags(write=True)
        
        NEW_GRID[cell] = CELL_STATE.FLAGGED
        
        NEW_GRID.setflags(write=False)

        # update
        return (Res_Code.FLAG_SUCESSFULLY_PLACED, self.update_grid(NEW_GRID))
    
    
    def remove_flag(self, cell:uint16) -> tuple[Res_Code, MinesweeperLogic]:
        # check if field is Forbidden
        if (self._isForbidden(cell)):
            return (Res_Code.TRIED_TO_UNFLAG_FORBIDDEN_CELL, self)
        
        # check if field is already reveald
        if (self.GRID[cell] in KNOWN_STATES):
            return (Res_Code.TRIED_TO_UNFLAG_REAVEALED_CELL, self)
        
        # check if field is already unflagged
        if (self.GRID[cell] == CELL_STATE.COVERED):
            return (Res_Code.TRIED_TO_UNFLAG_COVERED_CELL, self)
        
        # check if field is covered
        if (self.GRID[cell] != CELL_STATE.FLAGGED):
            return (Res_Code.TRIED_TO_UNFLAG_UNFLAGGED_CELL, self)
        
        # place flag
        NEW_GRID = self.GRID.copy()
        NEW_GRID.setflags(write=True)
        
        NEW_GRID[cell] = CELL_STATE.COVERED
        
        NEW_GRID.setflags(write=False)

        # update
        return (Res_Code.FLAG_SUCESSFULLY_REMOVED, self.update_grid(NEW_GRID))
    
    
    def get_cells_of_state(self, cells: Iterable, state: CELL_STATE):        
        # convert cells to ndarray
        cell_arr = np.array(list(cells))
        cell_arr.setflags(write=False)
        
        # create a mask of states
        mask = self.GRID[cell_arr] == state
        
        return cell_arr[mask]
    

    def update_grid(self, grid: np.ndarray[CELL_STATE]):
        # make sure
        NEW_GRID = grid.copy()
        
        # Create new Game, with the updated Grid
        newGame = MinesweeperLogic(
            COLS=self.COLS, 
            ROWS=self.ROWS, 
            MINE_FIELDS=self.MINE_FIELDS, 
            SEED=self.SEED, 
            LOWER_COL_RANGE=self.LOWER_COL_RANGE, 
            UPPER_COL_RANGE=self.UPPER_COL_RANGE,
            LOWER_ROW_RANGE=self.LOWER_ROW_RANGE,
            UPPER_ROW_RANGE=self.UPPER_ROW_RANGE, 
            MAP_BOUNDARIES_COLS=self.MAP_BOUNDARIES_COLS,
            MAP_BOUNDARIES_ROWS=self.MAP_BOUNDARIES_ROWS,
            GRID=NEW_GRID)
        
        # prevent modification of the new grid
        newGame.GRID.setflags(write=False)
        
        return newGame
    
    
    def convert_absolute_to_relative_index(self, cell: uint16):
        width = self.COLS
        row = (cell // self.MAP_BOUNDARIES_COLS) - 1
        col = (cell%self.MAP_BOUNDARIES_COLS) -1
        return col+row*width
    
    
    def convert_relative_coords_to_absolute_index(self, coord):
        width = self.MAP_BOUNDARIES_COLS
        row = coord[1]+1
        col = col[0]+1
        return col+row*width
    
        
    def convert_relative_to_absolute_index(self, cell: uint16):
        width = self.MAP_BOUNDARIES_COLS
        row = (cell // self.MAP_BOUNDARIES_COLS)
        col = (cell%self.MAP_BOUNDARIES_COLS)
        
        return col+row*width
    
    
    def __str__(self):
        """Returns a string representation of the Minesweeper game board."""
        if self.GRID is None:
            return "Minesweeper board not initialized."

        lines = []
        for r in range(self.MAP_BOUNDARIES_ROWS):
            row_cells = self.GRID[r * self.MAP_BOUNDARIES_COLS: (r + 1) * self.MAP_BOUNDARIES_COLS]
            row_str = " ".join(self._cell_repr(cell) for cell in row_cells)
            lines.append(row_str)

        return "\n".join(lines)
    
    
    def _cell_repr(self, cell):
        """Helper function to convert a cell state to a readable character."""
        if cell == CELL_STATE.FORBIDDEN:
            return "X"  # Forbidden cells
        elif cell == CELL_STATE.COVERED:
            return "#"  # Covered cells
        elif cell == CELL_STATE.ZERO:
            return "."  # Uncovered cells
        elif cell == CELL_STATE.FLAGGED:
            return "F"  # Flagged cells
        elif cell == CELL_STATE.MINE:
            return "*"  # Mines
        else:
            return str(int(cell))  # Default: Use numeric values