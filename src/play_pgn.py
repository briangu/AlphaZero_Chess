import chess
import torch
# import numpy as np
from alpha_net import ChessNet
from collections import deque
from util_pgn import *

def load_model(model_path):
    model = ChessNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    return model


def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(axis=0)

def get_model_move(board, model, move_history):
    current_board_encoded = encode_pychess_board(board)
    move_history.append(current_board_encoded)
    model_input = torch.stack(list(move_history))
    model_input = torch.tensor(model_input, dtype=torch.float32)

    # model_input = prepare_model_input(board, move_history)
    move_probs, value = model(model_input) # Modify according to your model's prediction method

    # Generate mask for legal moves
    legal_moves = board.legal_moves
    legal_move_mask = torch.zeros_like(move_probs, dtype=bool)

    for move in legal_moves:
        move_index = encode_move(board, move, tensor_out=False)  # Convert move to index
        legal_move_mask[move_index] = True

    # Apply mask
    masked_probs = torch.logical_and(move_probs, legal_move_mask)

    # Normalize probabilities
    normalized_probs = softmax(masked_probs)

    # Select move
    selected_move_index = torch.argmax(normalized_probs)
    selected_move = decode_move(selected_move_index, board)  # Convert index back to move

    return selected_move, value


def main():
    last_n_moves = 8
    model_path = sys.argv[1]
    model = load_model(model_path)
    board = chess.Board()

    move_history = deque(maxlen=last_n_moves)
    zero_state = torch.zeros_like(torch.tensor(encode_pychess_board(board), dtype=torch.float32))  # Adjust the shape/type as necessary
    for _ in range(last_n_moves):
        move_history.append(zero_state)

    while not board.is_game_over():
        print(board)

        if board.turn:
            # Model's turn
            move, value = get_model_move(board, model, move_history)
            print(value)
        else:
            # User's turn
            move = input("Enter your move: ")
            try:
                move = board.parse_san(move)
            except ValueError:
                print("Invalid move. Please try again.")
                continue

        board.push(move)

    print("Game over:", board.result())

if __name__ == "__main__":
    main()
