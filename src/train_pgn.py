import chess
import chess.pgn
import io
import copy
import torch
from torch.utils.data import Dataset
import encoder_decoder as ed
from torch.utils.data import DataLoader
import sys
from alpha_net import ChessNet, train

from chess_board import board as c_board


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


def process_game(pgn_text):
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    headers = game.headers
    Result = headers.get("Result")

    current_board = c_board()

    n = game.next()

    mate_score = 1 if Result == "1-0" else -1 if Result == "0-1" else 0

    while n is not None:
        initial_pos = n.move.from_square // 8, n.move.from_square % 8
        final_pos = n.move.to_square // 8, n.move.to_square % 8
        underpromote = convert_underpromotion(n.move.promotion)

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

        move_index = ed.encode_action(current_board, initial_pos, final_pos, underpromote=underpromote)

        # TODO: add support for providing a model that predicts the policy and value
        policy = torch.zeros(4672, 0.001, dtype=torch.float32)  # Assuming 4672 possible moves
        policy[move_index] = 1.0
        policy = policy / torch.sum(policy)

        value = normalize_stockfish_score(score) if score.isnumeric() else normalize_mate_score(score)

#        board_state = copy.deepcopy(ed.encode_board(current_board))
        board_state = torch.tensor(ed.encode_board(current_board))
        policy = torch.tensor(policy)
        value = torch.tensor(value)
        yield (board_state, policy, value)
        n = n.next()

    return dataset


class ChessPGNDataset(Dataset):
    def __init__(self, pgn_path, game_cnt):
        self.pgn_path = pgn_path
        self.game_cnt = game_cnt
        self.pgn_file = open(self.pgn_path, 'r')

    def __iter__(self):
        return self

    def __next__(self):
        # tobj = tqdm(enumerate(read_games(pgn_path)), total=game_cnt)
        for i, pgn_text in enumerate(read_games(pgn_path)):
            if pgn_text is None:
                break
            for state, policy, value in process_game(pgn_text):
                state_tensor = torch.FloatTensor(state)
                policy_tensor = torch.FloatTensor(policy)
                value_tensor = torch.FloatTensor(value)
                yield state_tensor, policy_tensor, value_tensor
        self.pgn_file.close()
        raise StopIteration

    def __len__(self):
        # Returning a large number as a placeholder since the exact length is unknown
        return self.game_cnt  # Adjust as needed


def train_chessnet(train_loader, net_to_train, save_as):
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    if net_to_train is not None:
        checkpoint = torch.load(net_to_train)
        net.load_state_dict(checkpoint['state_dict'])
    train(net,train_loader)
    torch.save({'state_dict': net.state_dict()}, save_as)


if __name__=="__main__":
    out_model_path = sys.argv[1]
#    pgn_path = sys.argv[2] if len(sys.argv) > 2 else "/data/lichess_db_standard_rated_2023-02.pgn"
    pgn_path = "/data/lichess_db_standard_rated_2023-02.pgn"
    game_cnt = 108201825
    # model_path = sys.argv[3] if len(sys.argv) > 3 else None
    model_path = None
    dataset = ChessPGNDataset(pgn_path)
    batch_size = 128  # You can adjust the batch size as needed
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    train_chessnet(train_loader, net_to_train=model_path,save_as=out_model_path)
