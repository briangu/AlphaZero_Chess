import os
import sqlite3
from contextlib import closing
import chess.pgn
from chess.engine import Cp, Mate, MateGiven
import multiprocessing as mp
from tqdm import tqdm
import sys
import multiprocessing
import io
import gc
import time

import numpy as np


def create_database(db_path):
    with sqlite3.connect(db_path) as conn:
        with closing(conn.cursor()) as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                black_elo INTEGER,
                white_elo INTEGER,
                termination TEXT,
                result TEXT
            )
            """)

            c.execute("""
            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY,
                game_id INTEGER,
                move TEXT,
                score INTEGER,
                mate INTEGER,
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
            """)
            conn.commit()


def writer_process(write_queue, db_path):
    create_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA temp_store = MEMORY")

        with closing(conn.cursor()) as c:
            while True:
                item = write_queue.get()  # Get data from the queue
                if item is None:
                    break  # Sentinel value received, time to exit

                if item['type'] == 'game':
                    # Insert game data
                    c.execute("INSERT INTO games (black_elo, white_elo, termination, result) VALUES (?, ?, ?, ?)",
                              (item['BlackElo'], item['WhiteElo'], item['Termination'], item['Result']))
                    game_id = c.lastrowid

                    # Insert move data
                    for move in item['moves']:
                        c.execute("INSERT INTO moves (game_id, move, score, mate) VALUES (?, ?, ?, ?)",
                                  (game_id, move[0], move[1], move[2]))

                if write_queue.qsize() == 0:  # Commit periodically or when queue is empty
                    conn.commit()


# def read_games(pgn_path, game_cnt):
#     with open(pgn_path) as pgn_file:
#         lines = []
#         send = False
#         for line in pgn_file:
#             lines.append(line)
#             if send:
#                 yield "".join(lines)
#                 lines = []
#                 send = False
#             elif line.startswith("1."):
#                 if 'eval' in line:
#                     send = True
#                 else:
#                     lines.clear()

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

def worker_process(game_queue, write_queue):
    while True:
        pgn_text = game_queue.get()
        if pgn_text is None:
            break
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))

            headers = game.headers
            BlackElo = headers.get("BlackElo")
            WhiteElo = headers.get("WhiteElo")
            Termination = headers.get("Termination")
            Result = headers.get("Result")

            n = game.next()
            moves = []
            while n is not None:
                move = str(n.move)
                e = n.eval()
                if e is None: # end of game
                    score, mate = None, 0
                else:
                    r = e.relative
                    if r is not None:
                        if r.is_mate():
                            score, mate = None, r.mate()
                        else:
                            score, mate = r.score(), None
                    else:
                        score, mate = None, None
                moves.append((move, score, mate))
                n = n.next()
        except Exception as e:
            print(pgn_text)
            import traceback
            traceback.print_exc(e)
            print(e)
            continue

        if len(moves) == 0:
            continue

        msg = {
            'type': 'game',
            'BlackElo': BlackElo,
            'WhiteElo': WhiteElo,
            'Termination': Termination,
            'Result': Result,
            'moves': moves
        }
        write_queue.put(msg)


def save_pgn_to_db(pgn_path, game_cnt, num_workers):
    game_queue = multiprocessing.Queue()
    write_queue = multiprocessing.Queue()  # Queue for the writer process

    # Start writer process
    writer = multiprocessing.Process(target=writer_process, args=(write_queue, "/data/evals.db"))
    writer.start()

    workers = [multiprocessing.Process(target=worker_process, args=(game_queue, write_queue)) for worker_id in range(num_workers)]
    for w in workers:
        w.start()

    game_threshold = 20000*num_workers
    write_threshold = 1000000
    tobj = tqdm(enumerate(read_games(pgn_path)), total=game_cnt)
    for i, raw_pgn in tobj:
        game_queue.put(raw_pgn)
        wq = write_queue.qsize()
        gq = game_queue.qsize()
        if gq > game_threshold or wq > write_threshold:
            time.sleep(0.1)
        if i % 100 == 0:
            tobj.set_postfix({"game_queue": gq, "write_queue": wq}, refresh=True)

    for _ in range(num_workers):
        game_queue.put(None)

    for w in workers:
        w.join()

    # Tell writer to finish
    write_queue.put(None)
    writer.join()  # Wait for writer to finish


# Example usage
pgn_path = "lichess_db_standard_rated_2023-02.pgn"
game_cnt = 108201825
num_workers = 32

save_pgn_to_db(pgn_path, game_cnt, num_workers)
