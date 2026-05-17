import berserk
import threading
import torch
import chess
import queue
import os
from huggingface_hub import HfApi
from engine import findBestMove, instantiateModel, WEIGHTS_PATH
from train import trainEngine
print("engine instantiated")

lockEval = threading.Lock()
boardState = queue.Queue()
model = instantiateModel()
optimizer = torch.optim.Adam(model.parameters(),lr=.001)

token = os.environ.get("LICHESS_TOKEN")

session = berserk.TokenSession(token)
client = berserk.Client(session)

def trainEngineLoop(boardState, model, optimizer):
    while True:
        state = boardState.get()
        with lockEval:
            trainEngine(state, model, optimizer)
            torch.save(model.state_dict(), "pytorch_model.bin")
            api = HfApi()
            api.upload_file(
            path_or_fileobj="pytorch_model.bin",
            path_in_repo="pytorch_model.bin",
            repo_id="ArindomP/chessbot",
            repo_type="model",
            commit_message=f"Updated weights after a lichess games"
            )

backgroundTraining = threading.Thread(target = trainEngineLoop, args = (boardState, model, optimizer), daemon=True)

class Game(threading.Thread):
    def __init__(self, client, game_id, queue, **kwargs):
        super().__init__(**kwargs)
        self.listBoardStates = []
        self.game_id = game_id
        self.client = client
        self.queue = queue

    def run(self):
        self.stream = self.client.bots.stream_game_state(self.game_id)
        self.current_state = next(self.stream)
        self.botWhite = self.current_state['white'].get('id') == 'chessbotap'
        board = chess.Board()
        moveList = self.current_state['state']['moves'].split()
        if moveList:
            for move in moveList:
                board.push_uci(move)
                self.listBoardStates.append(board.copy())
        botTurn = (board.turn == chess.WHITE and self.botWhite) or \
        (board.turn == chess.BLACK and not self.botWhite)
        if (botTurn):
            with lockEval:
                botMove = findBestMove(board, model)
            if botMove:
                self.client.bots.make_move(self.game_id, botMove.uci())
        for event in self.stream:
            self.listBoardStates = []
            if event['type'] == 'gameState':
                board = chess.Board()
                moveList = event['moves'].split()
                for move in moveList:
                    board.push_uci(move)
                    self.listBoardStates.append(board.copy())
                if board.is_game_over():
                    tempList = self.listBoardStates.copy()
                    self.queue.put(tempList)
                    self.listBoardStates.clear()
                    break
                botTurn = (board.turn == chess.WHITE and self.botWhite) or \
                (board.turn == chess.BLACK and not self.botWhite)
                if botTurn:
                    with lockEval:
                        bestMove = findBestMove(board,model)
                    if bestMove:
                        self.client.bots.make_move(self.game_id, bestMove.uci())

backgroundTraining.start()

try:
    for event in client.bots.stream_incoming_events():
        try:
            if event['type'] == 'challenge':
                client.bots.accept_challenge(event['challenge']['id'])
        except KeyError:
            pass
        if event['type'] == 'gameStart':
            game = Game(client, event['game']['id'], boardState, daemon=True)
            game.start()
except KeyboardInterrupt:
    print("stopped")