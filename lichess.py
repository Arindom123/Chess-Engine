import berserk

with open('./lichess.token') as f:
    token = f.read()

session = berserk.TokenSession(token)
client = berserk.Client(session)
#listen to lichess event stream with an infinite loop
#if statements to accept challenges and pass board to 
#findBestMove() and then pass the move back to Lichess
is_polite = True
for event in client.bots.stream_incoming_events():
    if event['type'] == 'challenge':
            client.bots.accept_challenge(event['id'])
    elif event['type'] == 'gameStart':
        game = Game(event['id'])
        game.start()
    class Game(threading.Thread):
    def __init__(self, client, game_id, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)

    def run(self):
        for event in self.stream:
            if event['type'] == 'gameState':
                self.handle_state_change(event)
            elif event['type'] == 'chatLine':
                self.handle_chat_line(event)

    def handle_state_change(self, game_state):
        pass

    def handle_chat_line(self, chat_line):
        pass
