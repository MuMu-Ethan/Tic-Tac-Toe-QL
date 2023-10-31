import numpy as np
import pygame

def Reset():
    global board
    board = []
    for i in range(9):
        board.append(0)

Reset()


def PrintBoard():

    print(" ")
    print(board[0], "|", board[1], "|", board[2])
    print("-----------")
    print(board[3], "|", board[4], "|", board[5])
    print("-----------")
    print(board[6], "|", board[7], "|", board[8])
    print(" ")


EPISODES = 100000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilon = 1
START_EPISILON_DECAY = 0
END_EPSILON_DECAY = EPISODES // 1.1
DECAY_VALUE = epsilon / (END_EPSILON_DECAY - START_EPISILON_DECAY)


def IsWinner(player):
    return (
        (board[0] == board[1] == board[2] == player) or
        (board[3] == board[4] == board[5] == player) or
        (board[6] == board[7] == board[8] == player) or
        (board[0] == board[3] == board[6] == player) or
        (board[1] == board[4] == board[7] == player) or
        (board[2] == board[5] == board[8] == player) or
        (board[0] == board[4] == board[8] == player) or
        (board[2] == board[4] == board[6] == player)
  )

train = input("Train? Choose y if you run it the first time (y/n):  ")

if train == "y":
    q_table1 = np.zeros([3] * 9 + [9])
    q_table2 = np.zeros([3] * 9 + [9])
else:
    q_table1 = np.load('q_table1.npy') 
    q_table2 = np.load('q_table2.npy')


def Action(q_table):
    possible_moves = [x for x, letter in enumerate(board) if letter == 0]
    moves = [-np.inf] * 9
    if len(possible_moves) != 0:
        for i in possible_moves:
            moves[i] = q_table[tuple(board)][i]
        if np.random.random() > epsilon:
            move = np.argmax(moves)
        else: 
            move = np.random.choice(possible_moves)
    else:
        move = -1
    return move


if train == "y":
    
    win = 0
    lose = 0
    tie = 0
    
    print("Starts Training")

    for episode in range(EPISODES):
        side = 0
        
        reward1 = 0
        reward2 = 0

        current_q1 = []
        next_max_q1 = []
        end_move1 = []
        board_state1 = []

        current_q2 = []
        next_max_q2 = []
        end_move2 = []
        board_state2 = []

        if (episode + 1) % 10000 == 0:
            print(episode + 1, "episodes")
        Reset() 
        while 0 in board and not IsWinner(1) and not IsWinner(2):  
            if side % 2 == 0:    
                
                action1 = Action(q_table1)
                
                end_move1.append(action1)
                state = tuple(board)
                current_q1.append(q_table1[state][action1])
                
                board_state1.append(state)
                
                board[action1] = 1
                new_state = tuple(board)
                next_max_q1.append(np.max(q_table1[new_state])) 
                
            if side % 2 != 0:     
                    
                action2 = Action(q_table2)
                
                end_move2.append(action2)
                state = tuple(board)
                current_q2.append(q_table2[state][action2])
                board_state2.append(state)
                board[action2] = 2
                new_state = tuple(board)
                next_max_q2.append(np.max(q_table2[new_state]))
            
            side += 1

        if IsWinner(1):
            reward1 = 1
            lose += 1
        elif IsWinner(2):
            reward1 = -1
            win += 1
        else:
            tie += 1
        
        reward2 = -1 * reward1

        for i in range(len(end_move1)):
            new_q = (1 - LEARNING_RATE) * current_q1[i] + (LEARNING_RATE * (reward1 + (DISCOUNT * next_max_q1[i])))
            q_table1[(board_state1[i])][end_move1[i]] = new_q 
        q_table1[(board_state1[-1])][end_move1[-1]] = reward1

        for i in range(len(end_move2)):
            new_q = (1 - LEARNING_RATE) * current_q2[i] + (LEARNING_RATE * (reward2 + (DISCOUNT * next_max_q1[i])))
            q_table2[(board_state2[i])][end_move2[i]] = new_q
        q_table2[(board_state2[-1])][end_move2[-1]] = reward2
        
        if START_EPISILON_DECAY <= episode <= END_EPSILON_DECAY:
            epsilon = epsilon - DECAY_VALUE

    print("")

    np.save('q_table1.npy', q_table1) 
    np.save('q_table2.npy', q_table2) 

    print("Training Finished")

    if train == "y":
        print("Progress Saved")

    print("")

    print("Results")
    print("Win:", win)
    print("Lose:", lose)
    print("Tie:", tie)

epsilon = 0

pygame.init()

WIN = pygame.display.set_mode((900, 900))
pygame.display.set_caption("Tic Tac Toe")

board_surface = pygame.image.load('graphics/board.png').convert_alpha()
playerX_surface = pygame.transform.scale(pygame.image.load('graphics/playerX.png').convert_alpha(), (960/4.5, 720/4.5))
playerO_surface = pygame.transform.scale(pygame.image.load('graphics/playerO.png').convert_alpha(), (960/4.5, 720/4.5))

def Font(x, y, size, text, color):
    font = pygame.font.Font('font/ARCADECLASSIC.ttf', size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center = (x, y))
    WIN.blit(text_surface, text_rect)

board_rect = board_surface.get_rect(center = (450, 450))

run = True

class Display:
    def __init__(self, x, y, surface):
        self.x = x
        self.y = y
        self.surface = surface

    def draw(self):
        WIN.blit(self.surface, (self.x, self.y))
    
positions = [(115, 159), (340, 159), (560, 159), (115, 372), (340, 372), (560, 372), (115, 594), (340, 594), (560, 594)]

ins = []
Reset()
while run:
    
    WIN.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            
            cont = False            
            player_x, player_y = pygame.mouse.get_pos()
            for i in range(3):
                if player_x > positions[i][0]:
                    final_player_x = positions[i][0]
    
            for i in range(3):
                if player_y > positions[3 * i][1]:
                    final_player_y = positions[3 * i][1]
            
            index = positions.index((final_player_x, final_player_y))
            pos_moves = [x for x, letter in enumerate(board) if letter == 0]

            if index in pos_moves:
                X = Display(final_player_x, final_player_y, playerX_surface)
                ins.append(X)
                board[index] = 1
                cont = True

            if IsWinner(1) or IsWinner(2) or 0 not in board:
                run = False
            
            if run and cont:    
                action = Action(q_table2)

                if action == -1:
                    run = False
                
                action_x, action_y = positions[action]
                board[action] = 2
                O = Display(action_x, action_y, playerO_surface)
                ins.append(O)

            if IsWinner(1) or IsWinner(2):
                run = False
    
    for n in ins:
        n.draw()
    
    WIN.blit(board_surface, board_rect)
    if IsWinner(1):
        Font(450, 450, 100, "You Win!", "green")
    elif IsWinner(2):
        Font(450, 450, 100, "You Lose!", "red")
    elif 0 not in board:
        Font(450, 450, 150, "Ties!", "orange")

    pygame.display.update()