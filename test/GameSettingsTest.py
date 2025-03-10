import unittest
from unittest import TestCase
from src.GameSettings import GameMap, GridSize


class MapSizeTest(TestCase):
    
    def test_if_beginner_size_fits_into_boundary(self):
        self.assertGreaterEqual(GameMap.BOUNDARIES.value[0]-2, GridSize.BEGINNER.value[0])
        self.assertGreaterEqual(GameMap.BOUNDARIES.value[1]-2, GridSize.BEGINNER.value[1])
    
    def test_if_intermediate_size_fits_into_boundary(self):
        self.assertGreaterEqual(GameMap.BOUNDARIES.value[0]-2, GridSize.INTERMEDIATE.value[0])
        self.assertGreaterEqual(GameMap.BOUNDARIES.value[1]-2, GridSize.INTERMEDIATE.value[1])
        
    def test_if_expert_size_fits_into_boundary(self):
        self.assertGreaterEqual(GameMap.BOUNDARIES.value[0]-2, GridSize.EXPERT.value[0])
        self.assertGreaterEqual(GameMap.BOUNDARIES.value[1]-2, GridSize.EXPERT.value[1])
        
if __name__ == "__main__":
    unittest.main()
