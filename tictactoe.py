import random
import time

class TicTacToe:
    def __init__(self):
        # Initialize the board as a 3x3 grid with empty spaces
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
        
    def print_board(self):
        """Display the current game board state"""
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')
            
    def print_board_nums(self):
        """Display the board with position numbers for reference"""
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')
    
    def available_moves(self):
        """Returns list of available moves (indexes of empty spaces)"""
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def empty_squares(self):
        """Returns True if there are empty squares on the board"""
        return ' ' in self.board
    
    def num_empty_squares(self):
        """Returns the number of empty squares"""
        return self.board.count(' ')
    
    def make_move(self, square, letter):
        """Places a letter on the specified square and returns True if valid"""
        if self.board[square] == ' ':
            self.board[square] = letter
            # Check if this move has created a winner
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False
    
    def winner(self, square, letter):
        """Checks if the last move has created a winner"""
        # Check row
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
            
        # Check column
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        
        # Check diagonals
        # Only need to check if move is on a diagonal
        if square % 2 == 0:
            # Check main diagonal (top-left to bottom-right)
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            # Check other diagonal (top-right to bottom-left)
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
                
        # If no winner
        return False

class Player:
    def __init__(self, letter):
        # letter is 'X' or 'O'
        self.letter = letter
        
    def get_move(self, game):
        pass

class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        
    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(f"{self.letter}'s turn. Input move (0-8): ")
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print("Invalid square. Try again.")
        return val

class AIPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        
    def get_move(self, game):
        # Add a slight delay to make the game feel more natural
        time.sleep(0.8)
        
        if len(game.available_moves()) == 9:
            # If it's the first move, randomly choose a position
            square = random.choice(game.available_moves())
        else:
            # Get the best move using the minimax algorithm
            square = self.minimax(game, self.letter)['position']
        return square
    
    def minimax(self, state, player):
        """
        Minimax algorithm implementation for the unbeatable AI
        Returns a dict with position and score of the optimal move
        """
        max_player = self.letter  # AI player
        other_player = 'O' if player == 'X' else 'X'
        
        # First, check if the previous move was a winner
        if state.current_winner == other_player:
            # We need to return position and score
            return {'position': None,
                    'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                                state.num_empty_squares() + 1)}
        elif not state.empty_squares():  # No empty squares (tie)
            return {'position': None, 'score': 0}
        
        # Initialize dictionary
        if player == max_player:
            best = {'position': None, 'score': float('-inf')}  # Maximizing player wants to maximize score
        else:
            best = {'position': None, 'score': float('inf')}  # Minimizing player wants to minimize score
        
        for possible_move in state.available_moves():
            # Step 1: make a move, try that spot
            state.make_move(possible_move, player)
            
            # Step 2: recurse using minimax to simulate a game after making that move
            sim_score = self.minimax(state, other_player)
            
            # Step 3: undo the move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move  # update the position value
            
            # Step 4: update the dictionary if necessary
            if player == max_player:  # Maximizing player
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:  # Minimizing player
                if sim_score['score'] < best['score']:
                    best = sim_score
                    
        return best

def play(game, x_player, o_player, print_game=True):
    """
    Main game loop function
    """
    if print_game:
        game.print_board_nums()
        print("")
    
    letter = 'X'  # starting letter
    
    # Continue the game as long as there are empty squares
    while game.empty_squares():
        # Get the move from the appropriate player
        if letter == 'X':
            square = x_player.get_move(game)
        else:
            square = o_player.get_move(game)
            
        # Make a move and update the game state
        if game.make_move(square, letter):
            if print_game:
                print(f"{letter} makes a move to square {square}")
                game.print_board()
                print("")
                
            # Check for winner
            if game.current_winner:
                if print_game:
                    print(f"{letter} wins!")
                return letter
                    
            # Switch players
            letter = 'O' if letter == 'X' else 'X'
            
    # Game ends in a tie
    if print_game:
        print("It's a tie!")

if __name__ == '__main__':
    while True:
        # Determine who goes first
        player_letter = ''
        while player_letter not in ['X', 'O']:
            player_letter = input("Do you want to be X or O? ").upper()
        
        if player_letter == 'X':
            human_player = HumanPlayer('X')
            ai_player = AIPlayer('O')
        else:
            human_player = HumanPlayer('O')
            ai_player = AIPlayer('X')
        
        # Create a new game
        t = TicTacToe()
        play(t, human_player, ai_player, print_game=True)
        
        # Ask if the player wants to play again
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            break
    
    print("Thanks for playing!")
