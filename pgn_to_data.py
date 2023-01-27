import os
import sys
import chess.pgn
import asyncio
import chess
import chess.engine
import multiprocessing
import numpy as np

Debug = False
Debug_Shorter_Analysis_List = False
Debug_Game = 'https://lichess.org/TDxYUKDD'
Debug_Pos = 2091174491

Batch_Size = 100  # Watch out for queue full for larger queue sizes
Min_Elo = 1000
Max_Elo = 2000
Valid_Time_Control = 300  # 5 min
Max_Clock_Incr = 5
Min_Clock = 60  # Moves discarded if the clock is below this value
Max_Abs_Score_Analysis = 600  # Max score to be considered for analysis
Max_Score_Cap = 600
Opening_Moves = 5  # Opening moves are not considered for analysis
Analysis_Moves = 20  # Number of qualifying moves for analysis
Max_CP_Loss = 600  # Centi Pawn Loss CAP http://talkchess.com/forum3/download/file.php?id=869
Train_Val_Ratio = 5  # Every one out of 4 data items used for validation

Train_Data_File = 'train_data.dat'
Val_Data_File = 'val_data.dat'
Train_Label_File = 'train_label.dat'
Val_Label_File = 'val_label.dat'

# engine, weights, nodes
lc0 = r"C:\ChessAI\lc0-v0.29.0-windows-cpu-dnnl\lc0.exe"
stockfish = r"C:\ChessAI\stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe"
# Sorted on the decreasing order of execution time
Analysis_List = ((stockfish, 'n/a', 10),
                 (stockfish, 'n/a', 12),
                 (lc0, r"C:\ChessAI\lc0-v0.29.0-windows-cpu-dnnl\791556.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1100.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1200.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1300.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1400.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1500.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1600.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1700.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1800.pb.gz", 1),
                 (lc0, r"C:\ChessAI\maia-weights\maia-1900.pb.gz", 1),
                 (stockfish, 'n/a', 0),
                 (stockfish, 'n/a', 1),
                 (stockfish, 'n/a', 2),
                 (stockfish, 'n/a', 3),
                 (stockfish, 'n/a', 4),
                 (stockfish, 'n/a', 5),
                 (stockfish, 'n/a', 6),
                 (stockfish, 'n/a', 7),
                 (stockfish, 'n/a', 8),
                 (stockfish, 'n/a', 9),
                 ('lichess', 'n/a', 'n/a'))
# https://chess.stackexchange.com/questions/29860/is-there-a-list-of-approximate-elo-ratings-for-each-stockfish-level
if Debug and Debug_Shorter_Analysis_List:
    Analysis_List = Analysis_List[:1]


def main(pgn_file, data_dir, target_items):
    dir_name = data_dir + '/' + str(target_items)
    os.makedirs(dir_name, exist_ok=True)

    file_handles = {'train_data': open(dir_name + '/' + Train_Data_File, 'wb'),
                    'val_data': open(dir_name + '/' + Val_Data_File, 'wb'),
                    'train_label': open(dir_name + '/' + Train_Label_File, 'wb'),
                    'val_label': open(dir_name + '/' + Val_Label_File, 'wb')}

    pgn = open(pgn_file)
    if Debug and Debug_Pos is not None:
        pgn.seek(Debug_Pos)

    # Process one batch of games at a time to limit the memory requirements
    for batch in range(0, target_items, Batch_Size):
        pgn_pos_list, elo_list_batch = get_pgn_pos_list(pgn, min(target_items, Batch_Size))
        print('pgn_pos_list[0]: ', pgn_pos_list[0])
        games = len(elo_list_batch)

        # initialize cp_loss_batch
        cp_loss_batch = []
        for i in range(games):
            cp_loss_batch.append([])
            for j in range(Analysis_Moves):
                cp_loss_batch[i].append([])

        # Analyze one batch of games
        tasks = [None] * len(Analysis_List)
        queue = multiprocessing.Queue()
        for idx, analysis in enumerate(Analysis_List):
            tasks[idx] = multiprocessing.Process(target=worker, args=(queue, idx, analysis, pgn_pos_list, pgn_file))
            tasks[idx].start()

        res_dict = {}  # dict for ordering the results from processes
        # Get the subprocesses outputs via the queue
        for idx in range(len(Analysis_List)):
            res = queue.get()
            res_dict[res[0]] = res[1:]

        # wait for the completion of the sub processes
        for idx in range(len(Analysis_List)):
            if tasks[idx].is_alive():
                tasks[idx].join()

        # Copy the subprocess outputs to the cp_loss_batch
        cp_loss_analysis  = None
        for idx in range(len(Analysis_List)):
            cp_loss_analysis = res_dict[idx][0]
            elo_list_analysis = res_dict[idx][1]
            if cp_loss_analysis is None:
                print('None  cp_loss_analysis, the subprocess=%d must have exited prematurely' % idx)
                print('Error: Batch %d of %d games failed' % (batch + Batch_Size, target_items))
                break
            else:
                assert (elo_list_batch == elo_list_analysis)
                for i in range(games):
                    for j in range(Analysis_Moves):
                        cp_loss_batch[i][j].append(cp_loss_analysis[i][j])

        if cp_loss_analysis is None:
            continue


        # Write cp_loss_batch to files
        for i in range(games):
            cp_loss_game = [cp_loss for moves in cp_loss_batch[i] for cp_loss in moves]
            if Debug:
                print('cp_loss_game:', np.array(cp_loss_game).reshape(Analysis_Moves, -1))
            if (i % Train_Val_Ratio) != 0:
                for j in range(Analysis_Moves * len(Analysis_List)):
                    file_handles['train_data'].write(cp_loss_game[j].to_bytes(4, byteorder='little', signed=False))
                file_handles['train_label'].write(elo_list_batch[i].to_bytes(4, byteorder='little', signed=False))
            else:
                for j in range(Analysis_Moves * len(Analysis_List)):
                    file_handles['val_data'].write(cp_loss_game[j].to_bytes(4, byteorder='little', signed=False))
                file_handles['val_label'].write(elo_list_batch[i].to_bytes(4, byteorder='little', signed=False))

        # Flush the files for each batch
        file_handles['train_data'].flush()
        file_handles['train_label'].flush()
        file_handles['val_data'].flush()
        file_handles['val_label'].flush()
        os.fsync(file_handles['train_data'])
        os.fsync(file_handles['train_label'])
        os.fsync(file_handles['val_data'])
        os.fsync(file_handles['val_label'])

        print('Gathering %d of %d games completed' % (batch + Batch_Size, target_items))

    file_handles['train_data'].close()
    file_handles['train_label'].close()
    file_handles['val_data'].close()
    file_handles['val_label'].close()


def worker(queue, idx, analysis, pgn_pos_list, pgn_file):
    pgn = open(pgn_file)
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    cp_loss_analysis, elo_list_analysis = asyncio.run(get_cp_loss_list(analysis, pgn, pgn_pos_list))
    queue.put((idx, cp_loss_analysis, elo_list_analysis))


def get_pgn_pos_list(pgn, Batch_Size):
    """
    Return offsets within the file for Batch_Size entries meeting the criteria.
    """
    pgn_pos_list = []
    elo_list = []

    next_pgn_pos = pgn.tell()
    while (game := chess.pgn.read_game(pgn)) is not None and len(elo_list) < Batch_Size:
        pgn_pos = next_pgn_pos
        next_pgn_pos = pgn.tell()

        # Discard Variant games
        for key in game.headers.keys():
            if "Variant" in key:
                continue

        # Ignore if both white and black ELOs are out of range
        try:
            white_elo = int(game.headers['WhiteElo'])
            black_elo = int(game.headers['BlackElo'])
        except ValueError:
            continue
        if (not Min_Elo <= black_elo <= Max_Elo) and (not Min_Elo <= white_elo <= Max_Elo):
            continue

        # Validate time control
        try:
            time_control = int(game.headers['TimeControl'].split('+')[0])
            time_incr = int(game.headers['TimeControl'].split('+')[1])
        except ValueError:
            continue
        if time_control != Valid_Time_Control or time_incr > Max_Clock_Incr:
            continue

        game_url = game.headers['Site']
        node = game
        clock = Min_Clock
        good_white_moves = 0
        good_black_moves = 0
        lichess_score = 0

        while (node := node.next()) is not None and \
                (good_white_moves < Analysis_Moves or good_black_moves < Analysis_Moves):
            if node.eval() is None:  # discard the pgn if no engine analysis available
                break

            if not node.board().outcome() is None:  # Stalemate, draw by repetition etc.
                break

            # lichess_score from the perspective of the player just moved
            prev_lichess_score = -lichess_score
            lichess_score = get_score_lichess(node, get_prev_turn(node))

            # Consider the move only if the time left on the clock is not too low
            prev_clock = clock  # clock of the player about to move
            clock = node.clock()  # clock of the player just moved
            if clock < Min_Clock:
                if prev_clock < Min_Clock:  # both players are low on clock
                    break
                else:  # only the player just moved is low on clock
                    continue

            if node.ply() < Opening_Moves * 2:  # Ignore opening moves
                continue

            if abs(prev_lichess_score) >= Max_Abs_Score_Analysis:  # Ignore completely winning and losing positions
                continue

            if get_prev_turn(node) == chess.WHITE:
                if Min_Elo <= white_elo <= Max_Elo:
                    good_white_moves += 1
                    if Debug and game_url == Debug_Game:
                        print('ply=%d, good_white_moves=%s' % (node.ply(), node.move))
            else:
                if Min_Elo <= black_elo <= Max_Elo:
                    good_black_moves += 1
                    if Debug and game_url == Debug_Game:
                        print('ply=%d, good_black_moves=%s' % (node.ply(), node.move))

        if good_white_moves >= Analysis_Moves or good_black_moves >= Analysis_Moves:
            pgn_pos_list.append(pgn_pos)
            if Debug and game_url == Debug_Game:
                print(game_url, good_white_moves, good_black_moves)

        if good_white_moves >= Analysis_Moves:
            elo_list.append(white_elo)
        if good_black_moves >= Analysis_Moves:
            elo_list.append(black_elo)

    return pgn_pos_list, elo_list


async def get_cp_loss_list(analysis, pgn, pgn_pos_list) -> []:
    # Set up the engine
    try:
        if 'stockfish' in analysis[0]:
            transport, engine = await chess.engine.popen_uci(analysis[0])
        elif 'lc0' in analysis[0]:
            transport, engine = await chess.engine.popen_uci(analysis[0])
            await engine.configure({"WeightsFile": analysis[1]})  # Configure weight file for the engine
        else:
            transport, engine = None, None
    except chess.engine.EngineTerminatedError:
        print('Exception: chess.engine.EngineTerminatedError')
        return None, None

    cp_loss_list = []
    elo_list = []

    for pos in pgn_pos_list:
        pgn.seek(pos)
        game = chess.pgn.read_game(pgn)
        game_url = game.headers['Site']

        white_elo = int(game.headers['WhiteElo'])
        black_elo = int(game.headers['BlackElo'])
        prev_score = 0
        score = 0
        lichess_score = 0
        prev_eng_move = None
        engine_move = None
        clock = Min_Clock
        good_white_moves = 0
        good_black_moves = 0
        white_losses = [0] * Analysis_Moves
        black_losses = [0] * Analysis_Moves

        node = game
        while (node := node.next()) is not None and \
                (good_white_moves < Analysis_Moves or good_black_moves < Analysis_Moves):

            if node.eval() is None:  # Checkmate has no eval
                break

            if not node.board().outcome() is None:  # Stalemate, draw by repetition etc.
                break

            # lichess_score from the perspective of the player just moved
            prev_lichess_score = -lichess_score
            lichess_score = get_score_lichess(node, get_prev_turn(node))

            if node.ply() >= Opening_Moves * 2 - 1:
                # Get scores from the perspective of the played just moved
                prev_score = -score  # Flip the perspective
                prev_eng_move = engine_move
                # print(game_url, node.ply(), node.move)
                # if node.board().outcome() is None:  # Game not ended
                if 'stockfish' in analysis[0]:
                    score, engine_move = await get_engine_info(engine, None, analysis[2], node,
                                                               perspective=get_prev_turn(node))
                elif 'lc0' in analysis[0]:
                    score, engine_move = await get_engine_info(engine, analysis[2], None, node,
                                                               perspective=get_prev_turn(node))
                else:
                    score = lichess_score
                    engine_move = 'na'
                # else:
                # score = lichess_score
                # engine_move = 'na'

            # Consider the move only if the time left on the clock is not too low
            prev_clock = clock  # clock of the player about to move
            clock = node.clock()  # clock of the player just moved
            if clock < Min_Clock:
                if prev_clock < Min_Clock:  # both players are low on clock
                    break
                else:  # only the player just moved is low on clock
                    continue

            if node.ply() < Opening_Moves * 2:  # Ignore opening moves
                continue

            if abs(prev_lichess_score) >= Max_Abs_Score_Analysis:  # Ignore completely winning and losing positions
                continue

            # Loss from the perspective of the player just moved
            if node.move == prev_eng_move:
                loss = 0
            else:
                gain = score - prev_score
                loss = max(0, -gain)
                loss = min(Max_CP_Loss, loss)

            if Debug and game_url == Debug_Game:
                print(game_url, node.ply(), get_prev_turn(node), clock, prev_eng_move, node.move,
                      prev_lichess_score, prev_score, score, loss)

            if get_prev_turn(node) == chess.WHITE:
                if Min_Elo <= white_elo <= Max_Elo:
                    if good_white_moves < Analysis_Moves:
                        white_losses[good_white_moves] = loss
                    good_white_moves += 1
                    if Debug and game_url == Debug_Game:
                        print('ply=%d, good_white_moves=%s' % (node.ply(), node.move))
            else:
                if Min_Elo <= black_elo <= Max_Elo:
                    if good_black_moves < Analysis_Moves:
                        black_losses[good_black_moves] = loss
                    good_black_moves += 1
                    if Debug and game_url == Debug_Game:
                        print('ply=%d, good_black_moves=%s' % (node.ply(), node.move))

        # append the losses for a given pgn
        if good_white_moves >= Analysis_Moves:
            cp_loss_list.append(white_losses)
            elo_list.append(white_elo)
        if good_black_moves >= Analysis_Moves:
            cp_loss_list.append(black_losses)
            elo_list.append(black_elo)
        # print(game_url, good_white_moves, good_black_moves)

        assert (good_white_moves >= Analysis_Moves or good_black_moves >= Analysis_Moves), \
            "pos=%d, url=%s, good_white_moves=%d, good_black_moves=%d" % \
            (pos, game_url, good_white_moves, good_black_moves)

    if analysis[0] != 'lichess':
        await engine.quit()
    return cp_loss_list, elo_list


def get_score_lichess(node, perspective):
    if (outcome := node.board().outcome()) is not None:  # Game ended
        if outcome.winner is None:
            score = 0
        elif outcome.winner == perspective:
            score = Max_Score_Cap
        else:
            score = -Max_Score_Cap
    elif node.eval().is_mate():  # Mate in the horizon
        if perspective == chess.WHITE:
            if node.eval().white().mate() > 0:
                score = Max_Score_Cap
            elif node.eval().white().mate() < 0:
                score = -Max_Score_Cap
            else:
                assert False, "Should not be here"
        else:
            if node.eval().black().mate() > 0:
                score = Max_Score_Cap
            elif node.eval().black().mate() < 0:
                score = -Max_Score_Cap
            else:
                assert False, "Should not be here"
    else:
        if perspective == chess.WHITE:
            score = node.eval().white().score()
        else:
            score = node.eval().black().score()

    score = min(score, Max_Score_Cap)
    score = max(score, -Max_Score_Cap)

    return score


async def get_engine_info(engine, engine_nodes, engine_depth, board_node, perspective) -> (int, str):
    if engine_nodes is not None:
        info = await engine.analyse(board_node.board(), chess.engine.Limit(nodes=engine_nodes))
    elif engine_depth is not None:
        info = await engine.analyse(board_node.board(), chess.engine.Limit(depth=engine_depth))
    else:
        assert False, 'Should not be here'
    engine_score = info['score']
    engine_move = info['pv'][0]

    score = 0
    if engine_score.is_mate():
        if perspective == chess.WHITE:
            if engine_score.white().mate() > 0:
                score = Max_Score_Cap
            elif engine_score.white().mate() < 0:
                score = -Max_Score_Cap
            else:
                assert False, "Should not be here"
        else:
            if engine_score.black().mate() > 0:
                score = Max_Score_Cap
            elif engine_score.black().mate() < 0:
                score = -Max_Score_Cap
            else:
                assert False, "Should not be here"
    else:
        if perspective == chess.WHITE:
            score = engine_score.white().score()
        else:
            score = engine_score.black().score()

    score = min(score, Max_Score_Cap)
    score = max(score, -Max_Score_Cap)

    return score, engine_move


def get_prev_turn(node):
    return chess.WHITE if node.turn() == chess.BLACK else chess.BLACK


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 pgn_to_data.py pgn-file, data-dir, number-positions")
        exit()
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
