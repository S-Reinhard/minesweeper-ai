import pytest
from src.GameSettings import GridSize, GridSize

def test_if_beginner_size_fits_into_boundary():
    """
    Test that the beginner grid size fits within the game map boundaries.
    The boundary must be aleast 2 cells wider and higher to account for a padding of FORBIDDEN cells on all 4 boarders of the playable area.
    """
    assert (GridSize.BOUNDARIES.value[0] >= GridSize.BEGINNER.value[0])  # Ensure width fits
    assert (GridSize.BOUNDARIES.value[1] >= GridSize.BEGINNER.value[1])  # Ensure height fits


def test_if_intermediate_size_fits_into_boundary():
    """
    Test that the intermediate grid size fits within the game map boundaries.
    The boundary must be aleast 2 cells wider and higher to account for a padding of FORBIDDEN cells on all 4 boarders of the playable area.
    """
    assert (GridSize.BOUNDARIES.value[0] >= GridSize.INTERMEDIATE.value[0])  # Ensure width fits
    assert (GridSize.BOUNDARIES.value[1] >= GridSize.INTERMEDIATE.value[1])  # Ensure height fits


def test_if_expert_size_fits_into_boundary():
    """
    Test that the expert grid size fits within the game map boundaries.
    The boundary must be aleast 2 cells wider and higher to account for a padding of FORBIDDEN cells on all 4 boarders of the playable area.
    """
    assert (GridSize.BOUNDARIES.value[0] >= GridSize.EXPERT.value[0])  # Ensure width fits
    assert (GridSize.BOUNDARIES.value[1] >= GridSize.EXPERT.value[1])  # Ensure height fits

