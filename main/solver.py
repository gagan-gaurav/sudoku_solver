# board = [
#     [7,8,0,4,0,0,1,2,0],
#     [6,0,0,0,7,5,0,0,9],
#     [0,0,0,6,0,1,0,7,8],
#     [0,0,7,0,4,0,2,6,0],
#     [0,0,1,0,5,0,9,3,0],
#     [9,0,4,0,6,0,0,0,5],
#     [0,7,0,3,0,0,0,1,2],
#     [1,2,0,0,0,7,4,0,0],
#     [0,4,9,2,0,6,0,0,7]
# ]
# board = [
#     [0,0,1,3,0,2,0,0,0],
#     [0,0,3,0,0,7,0,4,5],
#     [0,0,7,0,0,0,0,0,9],
#     [0,0,6,5,0,0,0,7,0],
#     [2,0,0,0,0,0,0,0,1],
#     [0,9,0,0,0,1,4,0,0],
#     [5,0,0,0,0,0,9,0,0],
#     [6,1,0,2,0,0,8,0,0],
#     [0,0,0,9,0,8,5,0,0]
# ]

# board = [
# 	[1,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,0],
# ]

# board = [
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,1],
# 	[1,1,1,1,1,1,1,1,0],
# ]
	

s = set()
def isValid(board):
	for i in range (0, 9):
		s.clear()
		for j in range (0, 9):
			if board[i][j] != 0:
				if board[i][j] in s:
					return False
				else:
					s.add(board[i][j])
	
	for j in range (0, 9):
		s.clear()
		for i in range (0, 9):
			if board[i][j] != 0:
				if board[i][j] in s:
					return False
				else:
					s.add(board[i][j])
	
	for i in range (0, 9, 3):
		for j in range (0, 9, 3):
			for x in range (i, i + 3):
				s.clear()
				for y in range (j, j + 3):
					if board[x][y] != 0:
						if board[x][y] in s:
							return False
						else:
							s.add(board[x][y])
	return True



def isSafe(grid, row, col, num):

	for x in range(9):
		if grid[row][x] == num:
			return False

	for x in range(9):
		if grid[x][col] == num:
			return False

	startRow = row - row % 3
	startCol = col - col % 3
	for i in range(3):
		for j in range(3):
			if grid[i + startRow][j + startCol] == num:
				return False
	return True


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")



def solve(i, j, board):
	if(j == 9):
		i += 1
		j = 0

	if i == 9 and j == 0:
		return True
	
	else:
		if board[i][j] != 0:
			return solve(i, j + 1, board)
		else:
			for k in range(1, 10):
				if isSafe(board, i, j, k):
					board[i][j] = k
					if solve(i, j + 1, board):
						return True
					board[i][j] = 0
			return
