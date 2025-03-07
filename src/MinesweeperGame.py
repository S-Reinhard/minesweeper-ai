from dataclasses import dataclass
from src.MinesweeperLogic import MinesweeperLogic, Res_Code
from src.GameSettings import Grid_Size
from collections import deque
from typing import Deque, Dict, Callable
from numpy import uint16
from enum import IntEnum

class Action_Type(IntEnum):
    """Enum defining possible action types in the game"""
    FLAG = 0
    UNFLAG = 1
    REVEAL = 2
    CHORD = 3


@dataclass(frozen=True)
class Action:
    """Represents an action taken in the game (immutable)"""
    ACTION_TYPE: Action_Type
    CELL: uint16

@dataclass(frozen=True)
class ActionHistoryEntry:
    """Stores an action along with its result code (immutable)"""
    ACTION: Action
    RESPONSE_CODE: Res_Code

class MinesweeperGame():
    """
    Manages the state, history, and actions of a Minesweeper game.

    This class maintains the game state, allowing actions such as flagging,
    revealing, and chording cells. It also tracks game history, supporting
    undo/redo functionality and time travel within game states.

    Attributes:
        game_history (Deque[MinesweeperLogic]): A deque storing snapshots of 
            the Minesweeper game states.
        action_history (Deque[ActionHistoryEntry]): A deque storing the actions 
            performed and their corresponding results.
        history_pointer (int): Points to the current state in the history.

    Properties:
        end_pointer (int): Returns the index of the last available history entry.
        history_size (int): Returns the number of stored history entries.
        current_game_state (MinesweeperLogic): Retrieves the current game state.

    Methods:
        in_past() -> bool:
            Checks if the history pointer is before the most recent state.
        in_present() -> bool:
            Checks if the history pointer is at the latest state.
        in_future() -> bool:
            Checks if the history pointer is beyond the latest recorded state.
        next() -> tuple[ActionHistoryEntry, MinesweeperLogic]:
            Returns the next action and game state if available.
        prev() -> tuple[ActionHistoryEntry, MinesweeperLogic]:
            Returns the previous action and game state if available.
        play_move(action: Action) -> tuple[Res_Code, MinesweeperLogic]:
            Executes an action and updates game history accordingly.
        undo() -> None:
            Moves the history pointer one step back if possible.
        redo() -> None:
            Moves the history pointer one step forward if possible.
        travel(offset: int) -> None:
            Moves the history pointer by a specified offset.
        jump(hist_index: int) -> None:
            Moves the history pointer to an absolute history index.
    """
    def __init__(self, cols=Grid_Size.BEGINNER[0], rows=Grid_Size.BEGINNER[1], seed=None):
        """Initialize a new Minesweeper game with a given grid size and optional seed."""
        init_state = MinesweeperLogic(COLS=cols, ROWS=rows, SEED=seed)
        
        # Maintain history of game states and actions
        self.game_history: Deque[MinesweeperLogic] = deque([init_state])
        self.action_history: Deque[ActionHistoryEntry] = deque([None])

        self.history_pointer = 0 # Tracks the current position in history
        
    @property
    def end_pointer(self) -> int:
        """Return the last valid index in the history."""
        return self.history_size - 1
    
    @property
    def history_size(self) -> int:
        """Return the size of the history. Ensure history consistency."""
        if len(self.game_history) != len(self.action_history):
            raise Exception("action history and state history are not matching")
        
        return len(self.game_history)
        
    def in_past(self) -> bool:
        """Check if the history pointer is in the past."""
        return self.history_pointer < self.end_pointer 
    

    def in_present(self) -> bool:
        """Check if the history pointer is at the most recent game state."""
        return self.history_pointer == self.end_pointer
    
    
    def in_future(self) -> bool:
        """Check if the history pointer is beyond the most recent state."""
        return self.history_pointer > self.end_pointer
    
    @property
    def current_game_state(self) -> MinesweeperLogic:
        """Return the current game state based on the history pointer."""
        return self.game_history[self.history_pointer]
    
    def next(self) -> tuple[ActionHistoryEntry, MinesweeperLogic]:
        """Return the next history entry if not already at the latest state."""
        if self.in_past():
            return (self.action_history[self.history_pointer + 1], self.game_history[self.history_pointer + 1])
        else:
            return tuple(None, None)
    
    def prev(self) -> tuple[ActionHistoryEntry, MinesweeperLogic]:
        """Return the previous history entry if possible."""
        if self.history_pointer > 0:
            return (self.action_history[self.history_pointer - 1], self.game_history[self.history_pointer - 1])
        else:
            return tuple(None, None)
    
    def _drop_future(self):
        """Remove all future history when a new action is taken in the past."""
        while (self.in_past()):
            self.pop_from_history()
            
    def _reset_pointer(self):
        """Reset the history pointer to the latest recorded state."""
        self.history_pointer = self.end_pointer
            
    
    def _add_to_history(self, action:ActionHistoryEntry, game_state: MinesweeperLogic):
        """Add a new state and action to the history."""
        # If in past, check if the future matches this action before overriding history
        if self.in_past():
            next_action, next_game = self.next()
            if action == next_action & game_state == next_game:
                self.redo()
                return
            else:
                self._drop_future()
                
        # Append the new state and action if in present
        if self.in_present():
            self.game_history.append(game_state)
            self.action_history.append(action)
            self.history_pointer = self.end_pointer
        
    
    def play_move(self, action: Action)-> tuple[Res_Code, MinesweeperLogic]:
        """Execute an action and update game history."""
        # Mapping of actions to their corresponding game logic methods
        actions: Dict[Action_Type, Callable[[uint16], tuple[Res_Code, MinesweeperLogic]]] = {
            Action_Type.FLAG: self.current_game_state.place_flag,
            Action_Type.UNFLAG: self.current_game_state.remove_flag,
            Action_Type.REVEAL: self.current_game_state.revealCell,
            Action_Type.CHORD: self.current_game_state.chord
        }
        
        # Execute the chosen action
        chosen_action = actions[action.ACTION_TYPE]
        res_code, new_state = chosen_action(action.CELL)
        
        # Create a history entry and store it
        actHistEntry = ActionHistoryEntry(action, res_code)
        self._add_to_history(actHistEntry, new_state)
        
        # Return the result of the move
        return (res_code, new_state)
    
    
    def undo(self):
        """Move the history pointer one step back if possible."""
        if (self.history_pointer > 0):
            self.history_pointer -= 1
            
    
    def redo(self):
        """Move the history pointer one step forward if possible."""
        if (self.history_pointer < self.end_pointer):
            self.history_pointer += 1
            
    
    def pop_from_history(self) -> tuple[ActionHistoryEntry, MinesweeperLogic]:
        """Remove the latest entry from history."""
        if (self.end_pointer > 0):
            self.action_history.pop()
            self.game_history.pop()
            
        if (self.in_future()):
            self._reset_pointer()
            
            
    def travel(self, offset: int) -> None:
        """Move the history pointer by a given offset."""
        new_pointer = self.history_pointer + offset
        
        if not (0 <= new_pointer <= self.end_pointer):
            raise IndexError("Offset out of bounds.")
        
        self.history_pointer = new_pointer

        
    def jump(self, hist_index: int) -> None:
        """Set the history pointer to an absolute index."""
        if hist_index < -self.history_size or hist_index > self.end_pointer:
            raise IndexError("Index out of bounds.")
        
        if hist_index < 0:
            self.history_pointer = self.history_size + hist_index  # Convert negative index to positive
        else:
            self.history_pointer = hist_index
