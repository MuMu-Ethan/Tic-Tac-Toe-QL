import numpy as np

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

def PlayerMove():
    run = True
    while run:
        move = input('Please select a position by typing a number 1~9: ')
        try:
            move = int(move)
            if move > 0 and move < 10:
                if board[move - 1] == 0:
                    run = False
                    board[move - 1] = 1
                else:
                    print('This postion is already occupied!')
            else:
                print('Please type a number in the range!')
        except:
            print('Please type a valid number!')


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
    for i in possible_moves:
        moves[i] = q_table[tuple(board)][i]
    if np.random.random() > epsilon:
        move = np.argmax(moves)
    else: 
        move = np.random.choice(possible_moves)

    return move

win = 0
lose = 0
tie = 0

print("Starts Training")

if train == "y":
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

    print("")
    print("############################################")
    print("")
    epsilon = 0

def Game():
    side = 0
    Reset()
    print("Welcome to Tic Tac Toe!")
    PrintBoard()
    while 0 in board and not IsWinner(1) and not IsWinner(2):
        if side % 2 == 0:
            PlayerMove()
        
        if side % 2 != 0:
            move = Action(q_table2)
            board[move] = 2
            PrintBoard()
            print("Computer plays in position", (move + 1))
        side += 1
    
    PrintBoard()
    
    if IsWinner(1):
        print("Congratulation! You won!")
    elif IsWinner(2):
        print("Oh no! You Lost!")
    else:
        print("Ties!")


while True:
    Game()
    a = input("Again? (y/n) ")
    if a == "y":
        pass
    else:
        break
