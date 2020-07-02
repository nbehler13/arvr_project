import math


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
        self.handvalues = []

    def new_round(self):
        self.handcards = []
        self.handvalues = []
        self.dealercards = []

    def set_predictions(self, boxes, labels):
        for card in labels:
            if card in self.available_cards:
                self.seen_cards.append(card)
                self.available_cards.remove(card)
        if len(self.handcards) == 0 or len(self.dealercards) == 0:
            self.set_hand_and_dealercards(boxes, labels)

    def set_hand_and_dealercards(self, boxes, labels):
        for i in range(len(boxes)):
            if boxes[i][0][0] < 200: # cards in the top of the frame belong to the dealer
                self.dealercards.append(labels[i])
                break
        distances = []
        for i in range(len(boxes)):
            if i+1 < len(boxes):
                distances.append(math.sqrt(math.pow(boxes[i+1][0][0]-boxes[i][0][0], 2) +
                                       math.pow(boxes[i+1][0][1]-boxes[i][0][1], 2)))
            else:
                distances.append(math.sqrt(math.pow(boxes[i][0][0]-boxes[0][0][0], 2) +
                                       math.pow(boxes[i][0][1]-boxes[0][0][1], 2)))
        for i in range(len(distances)):
            if distances[i] < 100: # TODO: find out with which distance two cards belong together
                if i+1 < len(labels):
                    self.handcards.append([labels[i], labels[i+1]])
                else:
                    self.handcards.append([labels[i], labels[0]])

        def predict_winning_loosing(player_id):
            hand = self.handcards[player_id]
            handvalue = 0
            for card in hand:
                for key in self.card_values:
                    if key in card:
                        handvalue += self.card_values[card]
                        break
            handvalue[player_id] = handvalue
            cards_smaller_21 = []
            cards_greater_21 = []
            for card in self.available_cards:
                for key in self.card_values:
                    if key in card:
                        if handvalue + self.card_values[key] <= 21:
                            cards_smaller_21.append(card)
                            break
                        else:
                            cards_greater_21.append(card)
                            break
            smaller_21 = len(cards_smaller_21)
            greater_21 = len(cards_greater_21)
            num_cards = len(self.available_cards) + len(self.seen_cards)
            win_chance = smaller_21/num_cards
            loose_chance = greater_21/num_cards
            return win_chance, loose_chance