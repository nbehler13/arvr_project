
class Player:
    def __init__(self, player_num, pos, name_pos):
        self.player_num = player_num
        self.player_name = "Dealer" if player_num == 0 else "Player "+str(player_num)
        self.handcards = []
        self.handvalue = 0
        self.show_pos = pos
        self.name_pos = name_pos
        self.handpositions = []
        self.win_chance = -1
        self.card_values = {'a': 1, 'k': 10, 'q': 10, 'j': 10, '10': 10, '9': 9, '8': 8, '7': 7,
                   '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}


    def get_new_cards(self, boxes, labels):
        #self.handcards = labels
        self.handcards = list(dict.fromkeys(labels))
        self.handvalue = 0
        for handcard in self.handcards:
            self.handvalue += self.card_values[handcard[:-1]]
        
        x_max = 0 # show win chance at upper right box
        y_min = 2000
        for box in boxes:
            if x_max < box[2]:
                x_max = box[2]
            if y_min > box[1]:
                y_min = box[1]
        self.show_pos = (int(x_max), int(y_min))


    def predict_winning(self, available_cards):
        if self.handvalue == 0: # if he didn't start playing, don't show win chance
            return
        amount_smaller_cards = 0
        for avail_card in available_cards:
            key = avail_card[:-1]
            value = self.card_values[key] # get card value
            if value + self.handvalue <= 21:
                amount_smaller_cards +=1 # add all cards that can be drawn
        #print("{}: {}".format(self.player_name, self.handcards))
        win_chance = amount_smaller_cards/len(available_cards)
        self.win_chance = round(win_chance*100)
        
