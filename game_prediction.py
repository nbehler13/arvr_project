import math
import numpy as np

class Predictor:
    available_cards = ['Ah', 'Kh', 'Qh', '10h', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h',
                       'Ac', 'Kc', 'Qc', '10c', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c',
                       'As', 'Ks', 'Qs', '10s', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s',
                       'Ad', 'Kd', 'Qd', '10d', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d']
    seen_cards = []
    card_values = {'A': 1, 'K': 10, 'Q': 10, 'J': 10, '10': 10, '9': 9, '8': 8, '7': 7,
                   '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

    def __init__(self):
        self.handcards = []
        self.dealercards = []
        self.handvalues = np.zeros(1)
        self.handpositions = []


    def new_round(self):
        self.handcards = []
        self.handvalues = []
        self.dealercards = []

    def update(self, boxes, labels):
        if not labels:
            return
        for card in labels:
            if card in self.available_cards:
                self.seen_cards.append(card)
                self.available_cards.remove(card)
        self.set_hand_and_dealercards(boxes, labels)

    def set_hand_and_dealercards(self, boxes, labels):
        for i in range(len(boxes)):
            if boxes[i][0] < 20: # TODO: find out until which pixel the dealer cards lay
                self.dealercards.append(labels[i])
                break
        distances = []
        for i in range(len(boxes)):
            if i+1 < len(boxes):
                distances.append(math.sqrt(math.pow(boxes[i+1][0]-boxes[i][0], 2) +
                                       math.pow(boxes[i+1][1]-boxes[i][1], 2)))
            else:
                distances.append(math.sqrt(math.pow(boxes[i][0]-boxes[0][0], 2) +
                                       math.pow(boxes[i][1]-boxes[0][1], 2)))
        print(distances)
        for i in range(len(distances)):
            if distances[i] < 100: # TODO: find out with which distance two cards belong in one hand
                if i+1 < len(labels):
                    self.handcards.append([labels[i], labels[i+1]])
                    self.handpositions.append([boxes[i][0], boxes[i][1], boxes[i+1][2], boxes[i+1][3]])
                else:
                    self.handcards.append([labels[i], labels[0]])
                    self.handpositions.append([boxes[i][0], boxes[i][1], boxes[0][2], boxes[0][3]])

    def predict_winning_loosing(self, player_id):
        if player_id >= len(self.handcards):
            print('player not found!')
            return -1, -1, []
        hand = self.handcards[player_id]
        handvalue = 0
        for card in hand:
            for key in self.card_values:
                if key in card:
                    handvalue += self.card_values[card[:-1]]
                    break
        self.handvalues[player_id] = handvalue
        #print(hand)
        #print(handvalue)
        cards_smaller_21 = []
        cards_greater_21 = []
        for card in self.available_cards:
            key = card[:-1]
            if key in card:
                if handvalue + self.card_values[key] <= 21:
                    cards_smaller_21.append(card)
                else:
                    cards_greater_21.append(card)
        smaller_21 = len(cards_smaller_21)
        greater_21 = len(cards_greater_21)
        num_cards = len(self.available_cards) + len(self.seen_cards)
        win_chance = round((smaller_21/num_cards)*100)
        loose_chance = round((greater_21/num_cards)*100)
        return win_chance, loose_chance, self.handpositions[player_id]