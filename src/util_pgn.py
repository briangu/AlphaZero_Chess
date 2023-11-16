import chess
import chess.pgn
import io
import torch
from torch.utils.data import Dataset
import encoder_decoder as ed
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import sys
from alpha_net import ChessNet, AlphaLoss
import os
from chess_board import board as c_board
from torch.utils.data import IterableDataset
import numpy as np
from collections import deque
# import time
import chess
import numpy as np


# Function to extract the game result
def get_game_result(game):
    result = game.headers["Result"]
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0


def read_games(pgn_path):
    game_buffer = io.StringIO()  # Efficient string buffer
    with open(pgn_path, 'r') as pgn_file:
        for line in pgn_file:
            if line.startswith("1."):
                if 'eval' in line:
                    game_buffer.write(line)
                    yield game_buffer.getvalue()  # Get the complete game string
                    game_buffer = io.StringIO()  # Reset the buffer for the next game
                else:
                    game_buffer = io.StringIO()  # Reset the buffer if not a valid game
            else:
                game_buffer.write(line)  # Keep appending lines to the current game buffer
    yield None


def normalize_stockfish_score(score, max_abs_score=1000):
    # Assuming max_abs_score is the maximum absolute score to be normalized
    if score is None:
        return 0
    # Clamp the scores to the maximum absolute score
    score = max(min(score, max_abs_score), -max_abs_score)
    # Normalize to [-1, 1]
    return score / max_abs_score

def normalize_mate_score(score):
    score = score[1:]
    return score / abs(score)

promotion_dict = {
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop"
}

def convert_underpromotion(promotion):
    return promotion_dict.get(promotion)


def convert_board(board):
    s = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == " ":
                s.append(".")
            else:
                s.append(board[i][j])
        s.append("\n")
    return "".join(s)


# compare the alphazero board with the python-chess board
def are_same(chess_board, pychess_board):
    for i in range(8):
        for j in range(8):
            piece = pychess_board.piece_at(8 * (7 - i) + j)
            piece = " " if piece is None else piece.symbol()
            #print(chess_board[i][j], piece)
            if chess_board[i][j] != piece:
                return False
    return True

# copy the python chess board to the alphazero board
def copy_board(pychess_board, chess_board):
    for i in range(8):
        for j in range(8):
            piece = pychess_board.piece_at(8 * (7 - i) + j)
            piece = " " if piece is None else piece.symbol()
            chess_board.current_board[i][j] = piece

def encode_pychess_board(board):
    """
    https://arxiv.org/pdf/2111.09259.pdf

    The first twelve 8 × 8 channels in z0 are binary, encoding the positions of the playing side
    and opposing side’s king, queen(s), rooks, bishops, knights and pawns respectively.

    It is followed by 8 x 8 binary channels representing the number of repetitions (for three-fold repetition draws),

    the side to play,
    and four binary channels for whether the player and opponent can still castle king and queenside.

    Finally, the last two channels are an irreversible move counter (for 50 move rule)
    and total move counter, both scaled down.
    """
    encoded = np.zeros([8, 8, 22]).astype(int)
    encoder_dict = {"R": 0, "N": 1, "B": 2, "Q": 3, "K": 4, "P": 5,
                    "r": 6, "n": 7, "b": 8, "q": 9, "k": 10, "p": 11}

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            if piece:
                encoded[i, j, encoder_dict[piece.symbol()]] = 1

    encoded[:, :, 12] = 1 if board.turn else 0
    encoded[:, :, 13] = 0 if board.has_kingside_castling_rights(chess.WHITE) else 1
    encoded[:, :, 14] = 0 if board.has_queenside_castling_rights(chess.WHITE) else 1
    encoded[:, :, 15] = 0 if board.has_kingside_castling_rights(chess.BLACK) else 1
    encoded[:, :, 16] = 0 if board.has_queenside_castling_rights(chess.BLACK) else 1
    encoded[:, :, 17] = board.fullmove_number / 200
    # For repetitions and no progress count, you may need additional logic
    # encoded[:, :, 18] = ...
    # encoded[:, :, 19] = ...
    encoded[:, :, 20] = board.halfmove_clock / 100
    if board.ep_square:
        ep_square = chess.square_file(board.ep_square), chess.square_rank(board.ep_square)
        encoded[ep_square[1], ep_square[0], 21] = 1

    return torch.tensor(encoded, dtype=torch.float32)


underpromotions = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

def encode_move(board, move, tensor_out=True):
    from_square = move.from_square
    to_square = move.to_square

    i, j = chess.square_file(from_square), chess.square_rank(from_square)
    x, y = chess.square_file(to_square), chess.square_rank(to_square)

    dx, dy = x - i, y - j

    promo = move.promotion
    if promo is not None and promo != chess.QUEEN:
        # Encoding underpromotion with 9 unique indices
        if i == x:
            file_offset = 0  # No file change
        elif i > x:
            file_offset = 1  # Left capture
        else:
            file_offset = 2  # Right capture
        # Calculate the index for underpromotion
        idx = 64 + 3 * file_offset + (promo - chess.KNIGHT) # exploit that KNIGHT = 2, BISHOP = 3, ROOK = 4
    elif board.piece_type_at(from_square) == chess.KNIGHT:
        if (x,y) == (i+2,j-1):
            idx = 56
        elif (x,y) == (i+2,j+1):
            idx = 57
        elif (x,y) == (i+1,j-2):
            idx = 58
        elif (x,y) == (i-1,j-2):
            idx = 59
        elif (x,y) == (i-2,j+1):
            idx = 60
        elif (x,y) == (i-2,j-1):
            idx = 61
        elif (x,y) == (i-1,j+2):
            idx = 62
        elif (x,y) == (i+1,j+2):
            idx = 63
    else:
        if dx != 0 and dy == 0: # north-south idx 0-13
            if dx < 0:
                idx = 7 + dx
            elif dx > 0:
                idx = 6 + dx
        if dx == 0 and dy != 0: # east-west idx 14-27
            if dy < 0:
                idx = 21 + dy
            elif dy > 0:
                idx = 20 + dy
        if dx == dy: # NW-SE idx 28-41
            if dx < 0:
                idx = 35 + dx
            if dx > 0:
                idx = 34 + dx
        if dx == -dy: # NE-SW idx 42-55
            if dx < 0:
                idx = 49 + dx
            if dx > 0:
                idx = 48 + dx

    # if tensor_out:
    encoded_move = torch.zeros((8, 8, 73), dtype=torch.float32)
    encoded_move[i, j, idx] = 1
    encoded_move = encoded_move.flatten()

    # return just the index into the flattened array
    return encoded_move if tensor_out else torch.where(encoded_move == 1)[0].item()


# def decode_move(encoded, board):
#     encoded_a = np.zeros([4672])
#     encoded_a[encoded] = 1
#     encoded_a = encoded_a.reshape(8, 8, 73)
#     a, b, c = np.where(encoded_a == 1)

#     for pos in zip(a, b, c):
#         i, j, k = pos
#         initial_square = chess.square(j, i)
#         final_square = None
#         promotion = None

#         # Decode the move based on the index k
#         if 0 <= k <= 55:
#             # Regular and knight moves
#             dx, dy = decode_regular_and_knight_moves(k)
#             final_square = chess.square(j + dx, i + dy)
#         elif 56 <= k <= 72:
#             # Underpromotion moves
#             final_square, promotion = decode_underpromotion_moves(k, i, j, board.turn)

#         # Auto-queen promotion for pawn reaching last rank without specified promotion
#         if board.piece_at(initial_square) == chess.Piece(chess.PAWN, board.turn) and chess.square_rank(final_square) in [0, 7] and promotion is None:
#             promotion = chess.QUEEN

#         if final_square is not None:
#             return chess.Move(initial_square, final_square, promotion)

#     raise ValueError("Invalid encoded move")

# def decode_regular_and_knight_moves(k):
#     if 0 <= k <= 13:
#         # North-south moves
#         return (k - 7, 0) if k < 7 else (k - 6, 0)
#     elif 14 <= k <= 27:
#         # East-west moves
#         return (0, k - 21) if k < 21 else (0, k - 20)
#     elif 28 <= k <= 41:
#         # NW-SE diagonal moves
#         dx = (k - 35) if k < 35 else (k - 34)
#         return (dx, dx)
#     elif 42 <= k <= 55:
#         # NE-SW diagonal moves
#         dx = (k - 49) if k < 49 else (k - 48)
#         return (dx, -dx)
#     elif 56 <= k <= 63:
#         # Knight moves
#         knight_moves = [(2, -1), (2, 1), (1, -2), (-1, -2), (-2, 1), (-2, -1), (-1, 2), (1, 2)]
#         return knight_moves[k - 56]

# def decode_underpromotion_moves(k, i, j, turn):
#     file_offset, promo = divmod(k - 64, 3)
#     promo_piece = [chess.ROOK, chess.KNIGHT, chess.BISHOP][promo]

#     dx = 0 if file_offset == 0 else (-1 if file_offset == 1 else 1)
#     dy = -1 if turn == chess.WHITE else 1

#     final_square = chess.square(j + dx, i + dy)
#     return final_square, promo_piece

promo_lookup = {
    'Q': chess.QUEEN,
    'q': chess.QUEEN,
    'R': chess.ROOK,
    'r': chess.ROOK,
    'B': chess.BISHOP,
    'b': chess.BISHOP,
    'N': chess.KNIGHT,
    'n': chess.KNIGHT,
}

def decode_move(encoded,board):
    encoded_a = np.zeros([4672]); encoded_a[encoded] = 1; encoded_a = encoded_a.reshape(8,8,73)
    a,b,c = np.where(encoded_a == 1); # i,j,k = i[0],j[0],k[0]
    # i_pos, f_pos, prom = [], [], []
    for pos in zip(a,b,c):
        i,j,k = pos
        initial_pos = (i,j)
        promoted = None
        if 0 <= k <= 13:
            dy = 0
            if k < 7:
                dx = k - 7
            else:
                dx = k - 6
            final_pos = (i + dx, j + dy)
        elif 14 <= k <= 27:
            dx = 0
            if k < 21:
                dy = k - 21
            else:
                dy = k - 20
            final_pos = (i + dx, j + dy)
        elif 28 <= k <= 41:
            if k < 35:
                dy = k - 35
            else:
                dy = k - 34
            dx = dy
            final_pos = (i + dx, j + dy)
        elif 42 <= k <= 55:
            if k < 49:
                dx = k - 49
            else:
                dx = k - 48
            dy = -dx
            final_pos = (i + dx, j + dy)
        elif 56 <= k <= 63:
            if k == 56:
                final_pos = (i+2,j-1)
            elif k == 57:
                final_pos = (i+2,j+1)
            elif k == 58:
                final_pos = (i+1,j-2)
            elif k == 59:
                final_pos = (i-1,j-2)
            elif k == 60:
                final_pos = (i-2,j+1)
            elif k == 61:
                final_pos = (i-2,j-1)
            elif k == 62:
                final_pos = (i-1,j+2)
            elif k == 63:
                final_pos = (i+1,j+2)
        else:
            if k == 64:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j)
                    promoted = "R"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j)
                    promoted = "r"
            if k == 65:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j)
                    promoted = "N"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j)
                    promoted = "n"
            if k == 66:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j)
                    promoted = "B"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j)
                    promoted = "b"
            if k == 67:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j-1)
                    promoted = "R"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j-1)
                    promoted = "r"
            if k == 68:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j-1)
                    promoted = "N"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j-1)
                    promoted = "n"
            if k == 69:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j-1)
                    promoted = "B"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j-1)
                    promoted = "b"
            if k == 70:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j+1)
                    promoted = "R"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j+1)
                    promoted = "r"
            if k == 71:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j+1)
                    promoted = "N"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j+1)
                    promoted = "n"
            if k == 72:
                if board.turn == chess.WHITE:
                    final_pos = (i-1,j+1)
                    promoted = "B"
                if board.turn == chess.BLACK:
                    final_pos = (i+1,j+1)
                    promoted = "b"
        piece = board.piece_at(chess.square(i,j))
        symbol = piece.symbol() if piece else " "
        if symbol in ["P","p"] and final_pos[0] in [0,7] and promoted == None: # auto-queen promotion for pawn
            if board.turn == chess.WHITE:
                promoted = "Q"
            else:
                promoted = "q"
        # i_pos.append(initial_pos); f_pos.append(final_pos), prom.append(promoted)
    from_square = chess.square(initial_pos[0], initial_pos[1])
    to_square = chess.square(final_pos[0], final_pos[1])
    return chess.Move(from_square, to_square, promotion=promo_lookup.get(promoted))


# def process_game(pgn_text):
#     game = chess.pgn.read_game(io.StringIO(pgn_text))

#     headers = game.headers
#     Result = headers.get("Result")

#     # current_board = c_board()

#     # last_move = game.move
#     value = 0.0
#     last_board = game.board()
#     n = game.next()

#     mate_score = 1 if Result == "1-0" else -1 if Result == "0-1" else 0

#     while n is not None:
#         # initial_pos = n.move.from_square // 8, n.move.from_square % 8
#         # final_pos = n.move.to_square // 8, n.move.to_square % 8
#         # initial_pos = 7 - chess.square_rank(n.move.from_square), chess.square_file(n.move.from_square)
#         # final_pos = 7 - chess.square_rank(n.move.to_square), chess.square_file(n.move.to_square)
#         # underpromote = convert_underpromotion(n.move.promotion)

#         e = n.eval()
#         if e is None: # end of game
#             score = mate_score
#         else:
#             r = e.relative
#             if r is not None:
#                 if r.is_mate():
#                     score = mate_score
#                 else:
#                     score = r.score()
#             else:
#                 score = mate_score

#         # # # print(current_board.current_board)
#         # if not are_same(current_board.current_board, last_board):
#         # #     print("not same:")
#         #     print(convert_board(current_board.current_board))
#         # #     print(last_move)
#         #     print(last_board)
#         # #     print(last_board.is_castling(n.move), n.move, initial_pos, final_pos, score)
#         #     raise RuntimeError("Boards are not same")
#         # move_index = ed.encode_action(current_board, initial_pos, final_pos, underpromote=underpromote)
#         policy = encode_move(last_board, n.move)

#         # TODO: add support for providing a model that predicts the policy and value
#         # policy = torch.zeros(4672, dtype=torch.float32) # + 0.001  # Assuming 4672 possible moves
#         # policy[move_index] = 1.0
#         # policy = policy / torch.sum(policy)

# #        board_state = copy.deepcopy(ed.encode_board(current_board))
#         # board_state = torch.tensor(ed.encode_board(current_board))
#         board_state = torch.tensor(encode_pychess_board(last_board), dtype=torch.float32)
#         # policy = torch.tensor(policy)
#         value = torch.tensor(value, dtype=torch.float32)
#         yield (board_state, policy, value)

#         value = normalize_stockfish_score(score) if isinstance(score, (int,float)) else normalize_mate_score(score)

#         # promoted_piece = chess.piece_symbol(last_move.promotion) if last_move and last_move.promotion is not None else "Q"

#         # if last_board.is_castling(n.move):
#         #     if last_board.is_kingside_castling(n.move):
#         #         current_board.castle("kingside", inplace=True)
#         #     else:
#         #         current_board.castle("queenside", inplace=True)
#         # else:
#         #     current_board.move_piece(initial_pos, final_pos, promoted_piece=promoted_piece)
#         # last_move = n.move
#         last_board = n.board()
#         # copy_board(n.board(), current_board)

#         n = n.next()

#     return dataset


def process_game(game, last_n_moves=8):
    headers = game.headers
    Result = headers.get("Result")

    value = 0.0
    last_board = game.board()
    n = game.next()

    mate_score = 1 if Result == "1-0" else -1 if Result == "0-1" else 0

    board_state_history = deque(maxlen=last_n_moves)
    zero_state = torch.zeros_like(encode_pychess_board(game.board()))  # Adjust the shape/type as necessary
    for _ in range(last_n_moves):
        board_state_history.append(zero_state)

    while n is not None:

        e = n.eval()
        if e is None: # end of game
            score = mate_score
        else:
            r = e.relative
            if r is not None:
                if r.is_mate():
                    score = mate_score
                else:
                    score = r.score()
            else:
                score = mate_score

        policy = encode_move(last_board, n.move)
        encoded_board = torch.tensor(encode_pychess_board(last_board), dtype=torch.float32)
        board_state_history.append(encoded_board)
        if last_board.fullmove_number < 10 and torch.rand(1) < 0.5:
            board_stack = torch.stack(list(board_state_history))
            value = torch.tensor(value, dtype=torch.float32)
            yield (board_stack, policy, value)

        value = normalize_stockfish_score(score) if isinstance(score, (int,float)) else normalize_mate_score(score)

        last_board = n.board()
        n = n.next()

    # return dataset


class ChessPGNDataset(IterableDataset):
    def __init__(self, pgn_path, game_cnt, last_n_moves=8, min_black_elo=1500, min_white_elo=1500):
        self.pgn_path = pgn_path
        self.game_cnt = game_cnt
        self.last_n_moves = last_n_moves
        self.min_black_elo = min_black_elo
        self.min_white_elo = min_white_elo

    def __iter__(self):
        for pgn_text in read_games(self.pgn_path):
            if pgn_text is None:
                return  # End of file
            game = chess.pgn.read_game(io.StringIO(pgn_text))

            headers = game.headers
            BlackElo = int(headers.get("BlackElo") or 1000)
            WhiteElo = int(headers.get("WhiteElo") or 1000)
            if BlackElo < self.min_black_elo or WhiteElo < self.min_white_elo:
                continue
            for state, policy, value in process_game(game, last_n_moves=self.last_n_moves):
                yield state, policy, value

    def __len__(self):
        return self.game_cnt
