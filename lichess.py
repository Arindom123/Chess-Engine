import berserk
import threading
import torch
import chess
import os
import queue
from engine import findBestMove
from engine import instantiateModel
from train import trainEngine

boardState = queue.Queue()
model = instantiateModel()
optimizer = torch.optim.Adam(model.parameters(),lr=.001)

with open('./lichess.token') as f:
    token = f.read()

session = berserk.TokenSession(token)
client = berserk.Client(session)

def trainEngineLoop(boardState, model, optimizer):
    while True:
            trainEngine(boardState.get(), model, optimizer)
            torch.save(model.state_dict(), "internalWeights.pth")
backgroundTraining = threading.Thread(target = trainEngineLoop, args = (boardState, model, optimizer))

class Game(threading.Thread):
    def __init__(self, client, game_id, queue, **kwargs):
        super().__init__(**kwargs)
        self.listBoardStates = []
        self.game_id = game_id
        self.client = client
        self.queue = queue
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        self.botWhite = self.current_state['white'].get('id') == 'chessbotap'

    def run(self):
        board = chess.Board()
        moveList = self.current_state['state']['moves'].split()
        if len(moveList) != 0:
            if moveList[0] != '':
                for move in moveList:
                    board.push_uci(move)
                    self.listBoardStates.append(board.copy())
        botTurn = (board.turn == chess.WHITE and self.botWhite) or \
        (board.turn == chess.BLACK and not self.botWhite)
        if (botTurn):
            botMove = findBestMove(board, model).uci()
            self.client.bots.make_move(self.game_id, botMove)
        for event in self.stream:
            if event['type'] == 'gameState':
                board = chess.Board()
                moveList = event['moves'].split()
                for move in moveList:
                    board.push_uci(move)
                self.listBoardStates.append(board.copy())
                if board.is_game_over():
                    self.queue.put(self.listBoardStates)
                    break
                botTurn = (board.turn == chess.WHITE and self.botWhite) or \
                (board.turn == chess.BLACK and not self.botWhite)
                if botTurn:
                    self.client.bots.make_move(self.game_id, findBestMove(board,model).uci())
                    self.listBoardStates.append(board.copy())
backgroundTraining.start()
for event in client.bots.stream_incoming_events():
    if event['type'] == 'challenge':
        client.bots.accept_challenge(event['challenge']['id'])
    if event['type'] == 'gameStart':
        listBoardStates = []
        game = Game(client, event['game']['id'], boardState)
        game.start()