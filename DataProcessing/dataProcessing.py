import pandas as pd
import os
import io
import re
import chess
import chess.pgn
from stockfish import Stockfish
import csv
import datetime
import berserk
import time
import json
import sqlite3
import multiprocessing
import requests
from itertools import cycle
from fake_useragent import UserAgent
import random
import subprocess

FREE_COUNTRIES = [
    "UK", "FR", "DE", "CH", "NO", "NL", "RO"
]
_last_country = [None]  # Mutable container for multiprocessing compatibility

def windscribe(action, location=None, max_retries=3):
    windscribe_cli_path = r"C:/Program Files/Windscribe/windscribe-cli.exe"
    if not os.path.exists(windscribe_cli_path):
        windscribe_cli_path = r"C:/Program Files (x86)/windscribe/windscribe-cli.exe"
    
    command = [windscribe_cli_path, action]
    if location:
        command.append(location)
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW  # Prevent console popups
            )
            print(f"Windscribe {action} success")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt+1} failed: {e.stderr}")
            if "already disconnected" in e.stderr.lower():
                return True  # Consider successful if already disconnected
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            time.sleep(1)
    
    print(f"Failed to execute {action} after {max_retries} attempts")
    return False

def windscribe_connect_new_country():
    global _last_country
    new_country = get_random_country(exclude=_last_country)
    
    # Ensure clean disconnect
    if not windscribe("disconnect"):
        # Force kill if normal disconnect fails
        os.system('taskkill /f /im windscribe-cli.exe 2>nul')
        time.sleep(2)
    
    # Handle connection
    if windscribe("connect", new_country):
        _last_country = new_country
        print(f"Connected to {new_country}")
        time.sleep(5)  # Allow VPN to stabilize
        return True
    return False

def get_random_country(exclude=None):
    choices = [c for c in FREE_COUNTRIES if c != exclude]
    return random.choice(choices) if choices else random.choice(FREE_COUNTRIES)

# Configure stockfish path
stockfish_path = './stockfish/stockfish.exe'
def get_stockfish():
    sf = Stockfish(stockfish_path)
    sf.set_depth(15)
    return sf

sf = get_stockfish()

if os.path.isfile(stockfish_path):
    print("The path is valid and points to a file.")
else:
    print("Invalid path or the file does not exist.")

sf.get_parameters()

def init_worker():
    global conn, cursor
    conn = sqlite3.connect('evaluations.db')
    cursor = conn.cursor()

def get_partial_fen(fen):
    parts = fen.split()
    return ' '.join(parts[:4])

def get_evaluation(fen, move_number):
    partial_fen = get_partial_fen(fen)
    cursor = conn.execute("SELECT eval_data FROM evaluations WHERE partial_fen = ? LIMIT 1", (partial_fen,))
    row = cursor.fetchone()

    if row:
        eval_data = json.loads(row[0])[0]['pvs'][0]
        if 'mate' in eval_data:
            if eval_data['mate'] > 0:
                return 10.0
            elif eval_data['mate'] < 0:
                return -10.0
            else:
                if move_number % 2 == 1:
                    return 10.0  # White wins
                else:
                    return -10.0  # Black wins
        elif 'cp' in eval_data:
            eval = eval_data['cp'] / 100.0
            return min(max(eval, -10.0), 10.0)
    else:
        sf.set_fen_position(fen)
        evaluation = sf.get_evaluation()
        if evaluation['type'] == 'mate':
            if evaluation['value'] > 0:
                return 10.0
            elif evaluation['value'] < 0:
                return -10.0
            else:
                if move_number % 2 == 1:
                    return 10.0
                else:
                    return -10.0
        else:
            eval = evaluation['value'] / 100.0
            return min(max(eval, -10.0), 10.0)

def get_top_moves(fen):
    partial_fen = get_partial_fen(fen)
    cursor = conn.execute("SELECT eval_data FROM evaluations WHERE partial_fen = ? LIMIT 1", (partial_fen,))
    row = cursor.fetchone()
    if row:
        eval_data = json.loads(row[0])
        top_moves = []
        seen_moves = set()
        for evaluation in eval_data:
            pvs = evaluation['pvs']
            for pv in pvs:
                line = pv['line'].split()
                if line:
                    move = line[0]
                    if move not in seen_moves:
                        seen_moves.add(move)
                        score = {'Mate': pv.get('mate'), 'Centipawn': pv.get('cp')}
                        top_moves.append({'Move': move, **score})
                        if len(top_moves) >= 6:
                            break
            if len(top_moves) >= 6:
                break
        return top_moves
    else:
        sf.set_fen_position(fen)
        return sf.get_top_moves(3)


def get_public_data_with_retry(username, retries=10, delay=60):
    ua = UserAgent()
    attempt = 0

    while attempt < retries:
        headers = {'User-Agent': ua.random}
        try:
            r = requests.get(f'https://lichess.org/api/user/{username}', headers=headers, timeout=15)
            if r.status_code == 429:
                print(f"429 Too Many Requests. Changing IP with Windscribe, retrying in {delay}s...")
                windscribe_connect_new_country()
                attempt += 1
                time.sleep(delay)
                continue
            elif r.status_code == 404:
                print(f"User {username} not found.")
                return None
            elif r.ok:
                return r.json()
            else:
                r.raise_for_status()
        except Exception as e:
            print(f"Request failed: {e}. Retrying after 5s.")
            attempt += 1
            time.sleep(5)
    raise Exception("Max retries exceeded.")

def check_account(username):
    session = berserk.TokenSession('lip_OrBrEgAITM24MhUSFanZ')
    client = berserk.Client(session=session)

    data = get_public_data_with_retry(username)
    if data is None:
        return False

    try:
        is_banned = data['tosViolation']
        if is_banned:
            return False
    except KeyError:
        pass

    try:
        created_at = data['createdAt']
        created_at_dt = datetime.datetime.fromtimestamp(created_at / 1000)
        account_age = (datetime.date.today() - created_at_dt.date()).days
        return account_age >= 365
    except KeyError:
        return False


book_move_prefixes = set()

with open('opening.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        moves_str = row[-1].strip()
        tokens = moves_str.split()
        moves = []
        for token in tokens:
            if re.match(r'^\d+\.', token):
                move = token.split('.', 1)[1]
                moves.append(move)
            else:
                moves.append(token)
        for i in range(1, len(moves) + 1):
            book_move_prefixes.add(tuple(moves[:i]))

def extract_format(pgn_header):
    event = pgn_header.get("Event", "").lower()
    if "bullet" in event:
        return "Bullet"
    elif "blitz" in event:
        return "Blitz"
    elif "rapid" in event:
        return "Rapid"
    elif "classical" in event:
        return "Classical"
    return None

def material_count(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    count = 0
    for piece_type in piece_values:
        count += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        count -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return count

def calculate_bci(board):
    mobility = weighted_mobility(board) / 50
    king_safety = evaluate_king_safety(board) / 10
    tactics = tactical_factors(board) / 5
    return 0.4 * mobility + 0.3 * king_safety + 0.3 * tactics

def weighted_mobility(board):
    score = 0
    for move in board.legal_moves:
        if board.is_capture(move):
            score += 2
        elif board.gives_check(move):
            score += 1.5
        else:
            score += 1
    return score / 20

def evaluate_king_safety(board):
    king_square = board.king(board.turn)
    shield_squares = _get_shield_squares(king_square)
    pawn_shield = sum(1 for sq in shield_squares if board.piece_at(sq) == chess.PAWN)
    castled = 1 if board.has_castling_rights(board.turn) else 0
    return pawn_shield + 2 * castled

def _get_shield_squares(king_square):
    file = chess.square_file(king_square)
    rank = chess.square_rank(king_square)

    shield_squares = []
    if rank < 7:
        shield_squares.append(chess.square(file, rank + 1))
        if file > 0:
            shield_squares.append(chess.square(file - 1, rank + 1))
        if file < 7:
            shield_squares.append(chess.square(file + 1, rank + 1))
    if rank > 0:
        shield_squares.append(chess.square(file, rank - 1))
        if file > 0:
            shield_squares.append(chess.square(file - 1, rank - 1))
        if file < 7:
            shield_squares.append(chess.square(file + 1, rank - 1))

    return shield_squares

def tactical_factors(board):
    hanging = sum(1 for square in board.piece_map() if not _is_defended(board, square))
    pins = sum(1 for square in board.piece_map() if board.is_pinned(board.turn, square))
    return (hanging + pins) / 5

def _is_defended(board, square):
    piece = board.piece_at(square)
    if piece is None:
        return False

    defended_squares = set()
    for attacker_square in board.pieces(piece.piece_type, piece.color):
        if attacker_square == square:
            continue
        defended_squares.update(board.attacks(attacker_square))

    return square in defended_squares

def categorize_position(fen):
    board = chess.Board(fen)
    piece_count = len(board.piece_map())
    major_pieces = sum(2 for piece in board.piece_map().values() if piece.symbol().upper() in 'QR')
    minor_pieces = sum(1 for piece in board.piece_map().values() if piece.symbol().upper() in 'BN')
    pawn_count = sum(1 for piece in board.piece_map().values() if piece.symbol().upper() == 'P')
    king_safety = all(board.piece_at(square) and board.piece_at(square).symbol().upper() == 'K' for square in [chess.E1, chess.E8])
    piece_activity = sum(1 for move in board.legal_moves)
    pawn_structure = sum(1 for square in chess.SQUARES if board.piece_at(square) and board.piece_at(square).symbol().upper() == 'P' and (square in chess.SQUARES[8:16] or square in chess.SQUARES[48:56]))
    developed_pieces = sum(1 for square in chess.SQUARES if board.piece_at(square) and board.piece_at(square).symbol().upper() in 'BNQR' and (square not in chess.SQUARES[0:16] and square not in chess.SQUARES[48:64]))

    if piece_count > 20 and pawn_structure > 4 and king_safety and developed_pieces < 4:
        return "Opening"
    elif (major_pieces + minor_pieces >= 5) or developed_pieces >= 4:
        return "Middlegame"
    else:
        return "Endgame"

def is_dumb_game(base_path, game_format, elo_range, game_id):
    format_folder = os.path.join(base_path, game_format.lower(), "filtered")
    filename = os.path.join(format_folder, f"filtered_game_{elo_range}.csv")
    
    if not os.path.exists(filename):
        return False
    
    df = pd.read_csv(filename)
    return game_id in df['game_id'].values

def add_to_dumb_games(base_path, game_format, elo_range, game_id):
    format_folder = os.path.join(base_path, game_format.lower(), "filtered")
    os.makedirs(format_folder, exist_ok=True)
    filename = os.path.join(format_folder, f"filtered_game_{elo_range}.csv")
    
    df = pd.DataFrame({'game_id': [game_id]})
    df.to_csv(filename, mode="a", header=not os.path.exists(filename), index=False)

def parse_reference_player_data(pgn_data, elo_range=(1975, 2025)):
    processed_games = []
    autoencoder_clock_times = []
    split_games = re.split(r'(?=\[Event)', pgn_data.strip())
    game_number = 0

    elo_low, elo_high = elo_range
    base_path = os.path.abspath("data")

    for game_text in split_games:
        pgn = io.StringIO(game_text)
        game = chess.pgn.read_game(pgn)
        if not game or not re.search(r'\[%clk \d+:\d+:\d+\]', game_text):
            continue

        headers = game.headers
        necessary_fields = ["Event", "Site", "Date", "Round", "White", "Black", "Result"]
        if not all(field in headers for field in necessary_fields):
            continue

        game_id = headers.get("Site", "").split("/")[-1]
        if is_dumb_game(base_path, "bullet", elo_range, game_id):  # Assuming bullet as default; adjust if needed
            continue
        
        white, black = headers["White"], headers["Black"]
        white_elo, black_elo = int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0))
        game_result = headers["Result"]
        time_control = headers.get("TimeControl", "0+0")
        base_time, increment = map(int, time_control.split('+'))
        game_format = extract_format(headers)
        if game_format is None:
            add_to_dumb_games(base_path, "bullet", elo_range, game_id)  
            continue

        game_format_text = game_format.lower()
        format_folder = os.path.join(base_path, game_format_text)
        filename = os.path.join(format_folder, f"autoencoders_data_{elo_low}_{elo_high}.csv")
        if os.path.exists(filename):
            existing_data = pd.read_csv(filename)
            if game_id in existing_data["Game ID"].values:
                add_to_dumb_games(base_path, game_format, elo_range, game_id)
                continue
            
        if not (elo_range[0] <= white_elo <= elo_range[1] and elo_range[0] <= black_elo <= elo_range[1]):
            add_to_dumb_games(base_path, game_format, elo_range, game_id)
            continue

        if not (check_account(white) and check_account(black)):
            add_to_dumb_games(base_path, game_format, elo_range, game_id)
            continue

        node = game
        move_number = 0
        prev_white_time = base_time
        prev_black_time = base_time
        move_sequence = []

        game_number += 1
        start_time = time.time()
        print(f"Processing game {game_number}...")

        while node.variations:
            move = node.variations[0]
            move_number += 1
            san = node.board().san(move.move)
            move_sequence.append(san)
            book_move = 1 if tuple(move_sequence) in book_move_prefixes else 0
            player = white if move_number % 2 == 1 else black
            player_elo = white_elo if move_number % 2 == 1 else black_elo
            opponent_elo = black_elo if move_number % 2 == 1 else white_elo
            clock_match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', node.comment)
            clock_time = (prev_white_time if move_number == 1 else
                         sum(int(x) * t for x, t in zip(clock_match.groups(), [3600, 60, 1])) if clock_match else None)
            if clock_time is None:
                continue

            fen_before = node.board().fen()
            phase = categorize_position(fen_before)
            top_moves = get_top_moves(fen_before)
            eval_before = get_evaluation(fen_before, move_number)
            top_move = top_moves[0]["Move"]
            equi_optimal = [m["Move"] for m in top_moves if m.get("Centipawn", None) == top_moves[0].get("Centipawn")]
            
            if not (elo_range[0] <= player_elo <= elo_range[1]):
                think_time = (prev_white_time if move_number % 2 == 1 else prev_black_time) - clock_time
                if move_number % 2 == 1:
                    prev_white_time = clock_time + increment
                else:
                    prev_black_time = clock_time + increment
                node = move
                get_evaluation(node.board().fen(), move_number)
                continue

            node = move
            fen_after = node.board().fen()
            eval_after = get_evaluation(fen_after, move_number)
            eval_diff = eval_before - eval_after
            centipawn_loss = abs(eval_diff)
            material_change = material_count(node.board()) - material_count(move.board())
            move_match = 1 if move.uci() == top_move else 0
            equal_value = 1 if move.uci() in equi_optimal else 0
            think_time = (prev_white_time if move_number % 2 == 1 else prev_black_time) - clock_time
            remaining_time = prev_white_time if move_number % 2 == 1 else prev_black_time
            if move_number % 2 == 1:
                prev_white_time = clock_time + increment
            else:
                prev_black_time = clock_time + increment

            processed_games.append({
                "Game ID": game_id, "Player": player, "Player Elo": player_elo, "Opponent Elo": opponent_elo,
                "White": white, "Black": black, "Move No.": move_number, "Book Move": book_move, "Move": move.move.uci(),
                "Game Format": game_format, "FEN Before": fen_before, "Elo Difference": abs(white_elo - black_elo),
                "Move Match (MM)": move_match, "Equal-Value (EV)": equal_value, "Centipawn Loss (AD)": centipawn_loss,
                "Remaining Time Before Move": remaining_time, "Time Spent": think_time,
                "Eval Before": eval_before, "Eval After": eval_after, "Eval Diff": eval_diff,
                "Board Complexity": calculate_bci(node.board()), "Material Change": material_change,
                "Phase": phase, "Game Result": game_result
            })
            
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Finished processing game {game_number} in {processing_time:.2f} seconds.")
        autoencoder_clock_times.append({"Game ID": game_id, "White Clock Time": prev_white_time, "Black Clock Time": prev_black_time})

    return pd.DataFrame(processed_games), autoencoder_clock_times

def parse_lstm_human_data(df_ref):
    time_thresholds = {"Bullet": 10, "Blitz": 60, "Rapid": 300, "Classical": 600}
    df_lstm = df_ref[["Game ID", "Move No.", "Book Move", "Move", "Eval Diff", "Move Match (MM)", "Equal-Value (EV)",
                      "Time Spent", "Board Complexity", "Game Format", "Remaining Time Before Move", "FEN Before"]].copy()
    df_lstm["Material Balance"] = df_ref["FEN Before"].apply(lambda fen: material_count(chess.Board(fen)))
    df_lstm["Critical Time Binary"] = df_lstm.apply(
        lambda row: 1 if row["Remaining Time Before Move"] <= time_thresholds.get(row["Game Format"], 0) else 0, axis=1)
    df_lstm["Volatility Score"] = df_lstm["Eval Diff"].abs()
    return df_lstm

def parse_autoencoders_data(df_ref, clocktimes):
    df_clock = pd.DataFrame(clocktimes)
    grouped = df_ref.groupby("Game ID").agg({
        "Time Spent": ["mean", "var"], "Move Match (MM)": "mean", "Equal-Value (EV)": "mean",
        "Centipawn Loss (AD)": ["mean", "sum", "std"], "Player Elo": "mean", "Material Change": "last",
        "Game Format": "first"
    }).reset_index()
    grouped.columns = ["Game ID", "Avg Time/Move", "Time Variance", "Avg MM", "Avg EV",
                      "Avg AD", "Total Accumulated Loss", "Volatility Score", "Avg Player Elo", "Material Balance Ends", "Game Format"]

    df_auto = grouped.merge(df_clock, on="Game ID")
    df_auto["Clock Difference"] = df_auto["White Clock Time"] - df_auto["Black Clock Time"]
    df_auto["Blunder Frequency"] = df_ref.groupby("Game ID").apply(lambda g: (g["Centipawn Loss (AD)"] > 100).mean()).reindex(df_auto["Game ID"]).values
    df_auto["Avg Scaled Difference"] = df_auto["Avg AD"] / df_auto["Avg Player Elo"]
    df_auto["Time Consistency Score"] = df_auto["Time Variance"].apply(lambda x: 1/x if x != 0 else float("inf"))
    return df_auto[["Game ID", "Avg Time/Move", "Time Variance", "Avg MM", "Avg EV", "Avg AD",
                    "Total Accumulated Loss", "Avg Scaled Difference", "Material Balance Ends",
                    "Clock Difference", "Blunder Frequency", "Volatility Score", "Time Consistency Score", "Game Format"]]


def save_dataframes(dfs, elo_range=(1975, 2025)):
    names = ["reference_player_data", "autoencoders_data", "lstm_human_data"]
    elo_low, elo_high = elo_range
    base_path = os.path.abspath("data")
    os.makedirs(base_path, exist_ok=True)

    for name, df in zip(names, dfs):
        if df is None or df.empty:
            continue
        if "Game Format" not in df.columns:
            continue
        else:
            for game_format in df["Game Format"].unique():
                format_folder = os.path.join(base_path, game_format.lower())
                os.makedirs(format_folder, exist_ok=True)
                filename = os.path.join(format_folder, f"{name}_{elo_low}_{elo_high}.csv")
                format_df = df[df["Game Format"] == game_format]
                format_df.to_csv(filename, mode="a", header=not os.path.exists(filename), index=False)
                print("Successful Save")

def process_chunk(chunk, elo_range=(1975, 2025)):
    df_ref, clocktimes = parse_reference_player_data(chunk, elo_range)
    if df_ref.empty:
        return None, None
    df_auto = parse_autoencoders_data(df_ref, clocktimes)
    df_lstm = parse_lstm_human_data(df_ref)
    
    return df_ref, df_auto, df_lstm

def worker(args):
    chunk, elo_range = args
    dfs = process_chunk(chunk, elo_range)
    if dfs[0] is not None:
        save_dataframes(dfs, elo_range)
    return dfs

# Modify load_games_to_exclude function
def load_games_to_exclude(filename):
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("game_id\n")
        return set()
    
    try:
        # Check if file is empty
        if os.path.getsize(filename) == 0:
            return set()
            
        df = pd.read_csv(filename)
        if df.empty:
            return set()
            
        # Remove duplicates
        df = df.drop_duplicates(subset=['game_id'])
        # Save cleaned data
        df.to_csv(filename, index=False)
        return set(df['game_id'].tolist())
    except pd.errors.EmptyDataError:
        return set()



def filter_games(split_games, excluded_game_ids):
    filtered_games = []
    for game in split_games:
        site_match = re.search(r'\[Site "([^"]+)"\]', game)
        if site_match:
            site_url = site_match.group(1)
            game_id = site_url.split("/")[-1]
            if game_id not in excluded_game_ids:
                filtered_games.append(game)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        else:
            # Optionally handle games without a Site header
            print("Game without Site header, skipping.")
            continue
    return filtered_games


if __name__ == '__main__':
    elo_range = (1975, 2025)
    chunk_size = 75
    results = []

    base_path = os.path.abspath("data")
    # Create necessary folders first
    format_folder = os.path.join(base_path, "bullet", "filtered")
    os.makedirs(format_folder, exist_ok=True)  # Ensure folder exists
    
    excluded_games_filename = os.path.join(format_folder, f"filtered_game_{elo_range}.csv")
    excluded_game_ids = load_games_to_exclude(excluded_games_filename)

    with open("2000.pgn", "r") as f:
        pgn_data = f.read()
        split_games = re.split(r'(?=\[Event)', pgn_data.strip())

        split_games = filter_games(split_games, excluded_game_ids)

        pool = multiprocessing.Pool(processes=4, initializer=init_worker)
        chunks = ["\n\n".join(split_games[i:i + chunk_size]) for i in range(0, len(split_games), chunk_size)]
        results = pool.map(worker, [(chunk, elo_range) for chunk in chunks])
        pool.close()
        pool.join()