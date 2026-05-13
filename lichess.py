import berserk
import threading
import torch
import chess
from engine import findBestMove
from engine import instantiateModel

model = instantiateModel()
optimizer = torch.optim.Adam(model.parameters(),lr=.001)

with open('./lichess.token') as f:
    token = f.read()

session = berserk.TokenSession(token)
client = berserk.Client(session)

class Game(threading.Thread):
    def __init__(self, client, game_id, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
    
    def run(self):
        gameState = self.current_state
        board = chess.Board()
        moveList = event['moves'].split()
        for moves in moveList:
            board.push_uci(moves)
        for event in self.stream:
            if board.turn and gameState['white']['id'] == 'chessbotap':
                if not board.is_game_over():
                    client.bots.make_move(self.game_id, findBestMove(board,model).uci())
            if event['type'] == 'gameState':
                if not board.is_game_over():
                    if not board.turn and gameState['black']['id'] == 'chessbotap' or 
                        client.bots.make_move(self.game_id, findBestMove(board,model).uci())

for event in client.bots.stream_incoming_events():
    if event['type'] == 'challenge':
        client.bots.accept_challenge(event['challenge']['id'])
    elif event['type'] == 'gameStart':
        game = Game(client, event['game']['id'])
        game.start()