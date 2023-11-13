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


def process_game(pgn_text):
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    headers = game.headers
    Result = headers.get("Result")

    current_board = c_board()

    # last_move = game.move
    # last_board = game.board()
    n = game.next()

    mate_score = 1 if Result == "1-0" else -1 if Result == "0-1" else 0

    while n is not None:
        # initial_pos = n.move.from_square // 8, n.move.from_square % 8
        # final_pos = n.move.to_square // 8, n.move.to_square % 8
        initial_pos = 7 - chess.square_rank(n.move.from_square), chess.square_file(n.move.from_square)
        final_pos = 7 - chess.square_rank(n.move.to_square), chess.square_file(n.move.to_square)
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

        # # print(current_board.current_board)
        if not are_same(current_board.current_board, last_board):
        #     print("not same:")
            print(convert_board(current_board.current_board))
        #     print(last_move)
            print(last_board)
        #     print(last_board.is_castling(n.move), n.move, initial_pos, final_pos, score)
            raise RuntimeError("Boards are not same")
        move_index = ed.encode_action(current_board, initial_pos, final_pos, underpromote=underpromote)

        # TODO: add support for providing a model that predicts the policy and value
        policy = torch.zeros(4672, dtype=torch.float32) + 0.001  # Assuming 4672 possible moves
        policy[move_index] = 1.0
        policy = policy / torch.sum(policy)

        value = normalize_stockfish_score(score) if isinstance(score, (int,float)) else normalize_mate_score(score)

#        board_state = copy.deepcopy(ed.encode_board(current_board))
        board_state = torch.tensor(ed.encode_board(current_board))
        # policy = torch.tensor(policy)
        value = torch.tensor(value)
        yield (board_state, policy, value)

        # promoted_piece = chess.piece_symbol(last_move.promotion) if last_move and last_move.promotion is not None else "Q"

        # if last_board.is_castling(n.move):
        #     if last_board.is_kingside_castling(n.move):
        #         current_board.castle("kingside", inplace=True)
        #     else:
        #         current_board.castle("queenside", inplace=True)
        # # elif n.move.promotion is not None:
        #     # current_board.move_piece(initial_pos, final_pos, promoted_piece=promoted_piece)
        # else:
        #     current_board.move_piece(initial_pos, final_pos, promoted_piece=promoted_piece)
        # last_move = n.move
        last_board = n.board()
        copy_board(n.board(), current_board)
        n = n.next()

    return dataset


class ChessPGNDataset(IterableDataset):
    def __init__(self, pgn_path, game_cnt):
        self.pgn_path = pgn_path
        self.game_cnt = game_cnt

    def __iter__(self):
        for pgn_text in read_games(self.pgn_path):
            if pgn_text is None:
                return  # End of file
            for state, policy, value in process_game(pgn_text):
                # state_tensor = torch.FloatTensor(state)
                # policy_tensor = torch.FloatTensor(policy)
                # value_tensor = torch.FloatTensor([value])  # Ensure value is a tensor
                yield state, policy, value

    def __len__(self):
        return self.game_cnt


def train(net, train_loader, epoch_start=0, epoch_stop=20, cpu=0):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)

    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()
            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (os.getpid(), epoch + 1, (i + 1)*30, len(train_loader.dataset), total_loss/10))
                print("Policy:", policy[0].argmax().item(), policy_pred[0].argmax().item())
                print("Value:", value[0].item(), value_pred[0,0].item())
                losses_per_batch.append(total_loss/10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if len(losses_per_epoch) > 100:
            if abs(sum(losses_per_epoch[-4:-1])/3 - sum(losses_per_epoch[-16:-13])/3) <= 0.01:
                break
        scheduler.step()


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
    dataset = ChessPGNDataset(pgn_path, game_cnt)
    batch_size = 128  # You can adjust the batch size as needed
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    train_chessnet(train_loader, net_to_train=model_path,save_as=out_model_path)
