import math

# Initialize board
board = [" " for _ in range(9)]

# Print the board
def print_board():
    print("\n")
    for i in range(3):
        print("|".join(board[i*3:(i+1)*3]))
        if i < 2:
            print("-----")
    print("\n")

# Check if the game is over and return the winner
def check_winner(board):
    win_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]
    for (i, j, k) in win_combinations:
        if board[i] == board[j] == board[k] and board[i] != " ":
            return board[i]
    if " " not in board:
        return "Draw"
    return None

# Get available moves
def available_moves(board):
    return [i for i in range(len(board)) if board[i] == " "]

# Minimax algorithm with Alpha-Beta pruning
def minimax(board, depth, is_maximizing, alpha, beta):
    result = check_winner(board)
    if result == "O":
        return 1
    elif result == "X":
        return -1
    elif result == "Draw":
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for move in available_moves(board):
            board[move] = "O"
            eval = minimax(board, depth + 1, False, alpha, beta)
            board[move] = " "
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in available_moves(board):
            board[move] = "X"
            eval = minimax(board, depth + 1, True, alpha, beta)
            board[move] = " "
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# AI move
def ai_move():
    best_score = -math.inf
    best_move = None
    for move in available_moves(board):
        board[move] = "O"
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[move] = " "
        if score > best_score:
            best_score = score
            best_move = move
    board[best_move] = "O"

# Main game loop
def play_game():
    print("Welcome to Tic-Tac-Toe (You: X | AI: O)")
    print_board()

    while True:
        # Human move
        while True:
            try:
                move = int(input("Enter your move (0-8): "))
                if board[move] == " ":
                    board[move] = "X"
                    break
                else:
                    print("Cell already taken. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter a number between 0 and 8.")

        print_board()
        result = check_winner(board)
        if result:
            if result == "Draw":
                print("It's a Draw!")
            else:
                print(f"{result} wins!")
            break

        # AI move
        ai_move()
        print("AI has played:")
        print_board()
        result = check_winner(board)
        if result:
            if result == "Draw":
                print("It's a Draw!")
            else:
                print(f"{result} wins!")
            break

# Start the game
if __name__ == "__main__":
    play_game()
