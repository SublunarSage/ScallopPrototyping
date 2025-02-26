import scallopy
import numpy as np

# Initialize Scallopy context
ctx = scallopy.ScallopContext(provenance="unit")
ctx.import_file("scalloptest1/sudoku.scl")

# Function to solve Sudoku
def solve_sudoku(grid):
    # Add given clues to the context
    given_facts = [(r, c, grid[r][c]) for r in range(9) for c in range(9) if grid[r][c] != 0]
    ctx.add_relation("given", given_facts)

    # Run the program
    ctx.run()

    # Retrieve the solution
    solution = ctx.relation("solution").as_numpy()
    solved_grid = np.zeros((9, 9), dtype=int)
    for (r, c, n) in solution:
        solved_grid[r, c] = n
    return solved_grid

# Example usage
if __name__ == "__main__":
    # Partially filled Sudoku grid (0 represents empty cells)
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    solution = solve_sudoku(puzzle)
    print("Solved Sudoku Grid:")
    print(solution)
