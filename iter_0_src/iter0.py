# 0: Nothing in hand; not a recognized poker hand 
# 1: One pair; one pair of equal ranks within five cards
# 2: Two pairs; two pairs of equal ranks within five cards
# 3: Three of a kind; three equal ranks within five cards
# 4: Straight; five cards, sequentially ranked with no gaps
# 5: Flush; five cards with the same suit
# 6: Full house; pair + different rank three of a kind
# 7: Four of a kind; four equal ranks within five cards
# 8: Straight flush; straight + flush
# 9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

import numpy as np
import random as rand

def genTups(num):
    xc = range(1, num)
    yc = genGuess(num)
    return np.array(np.vstack((xc, yc)).T, dtype = int)

def genGuess(num):
    rawguess = [rand.randint(1, 2598960) for x in range(1, num)]
    return [clasify(x) for x in rawguess]

def clasify(x):
    if x <= 4:
        return 9
    elif x <= 40:
        return 8
    elif x <= 6664:
        return 7
    elif x <= 4408:
        return 6
    elif x <= 9516:
        return 5
    elif x <= 197166:
        return 4
    elif x <= 746628:
        return 3
    elif x <= 198180:
        return 2
    elif x <= 1296420:
        return 1
    else:
        return 0


if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    test = np.genfromtxt('test.csv', delimiter = ',')

    sz = test.shape[0]
    ans = genTups(sz)

    np.savetxt("resp0.csv", ans, fmt = '%d', header = 'id,hand', comments = '', delimiter=",")