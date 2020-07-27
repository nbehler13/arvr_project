from player import Player
import cv2
import numpy as np
from collections import Counter

class GameManager:

    def __init__(self, VID_WIDTH, VID_HEIGHT):
        self.num_players = 4
        self.players = []
        self.half_width = VID_WIDTH // 2
        self.half_height = VID_HEIGHT // 2
        self.card_buffer = {}
        self.buffer_counter = 0
        self.card_margin = 5 # distance card refers to same box in next frame 
        self.new_round_counter = 0
        self.show_new_round = True

        for i in range(self.num_players):
            self.players.append(Player(i)) # instance of player 0, player 1, ..., player num_players

        self.available_cards = ['ah', 'kh', 'qh', 'jh', '10h', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h',
                                'ac', 'kc', 'qc', 'jc', '10c', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c',
                                'as', 'ks', 'qs', 'js', '10s', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s',
                                'ad', 'kd', 'qd', 'jd', '10d', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d']

    def reset_round(self):
        self.card_buffer = {}
        self.buffer_counter = 0
        self.new_round_counter = 0
        self.show_new_round = True
        self.available_cards = ['ah', 'kh', 'qh', 'jh', '10h', '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h',
                                'ac', 'kc', 'qc', 'jc', '10c', '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c',
                                'as', 'ks', 'qs', 'js', '10s', '9s', '8s', '7s', '6s', '5s', '4s', '3s', '2s',
                                'ad', 'kd', 'qd', 'jd', '10d', '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d']


    def get_player_to_box(self, box):
        x = box[0] + (box[2] - box[0])/2 # Width direction
        y = box[1] + (box[3] - box[1])/2 # Height direction
        player_num = 0 # in which player box is the dectected card?
        if x < self.half_width and y < self.half_height: # upper left
            player_num = 0
        elif x > self.half_width and y < self.half_height: # upper right
            player_num = 1
        elif x > self.half_width and y > self.half_height: # bottom right
            player_num = 2
        elif x < self.half_width and y > self.half_height: # bottom_ left
            player_num = 3
        return player_num

    
    def buffer_cards(self, boxes, labels, confis):
        new_boxes = []
        new_labels = []

        for i in range(len(labels)):
            if confis[i] < 0.85:
                continue
            box = boxes[i]
            x = box[0]# + (box[2] - box[0])/2 # Width direction
            y = box[1]# + (box[3] - box[1])/2 # Height direction

            buffer_key = -1
            for key in self.card_buffer.keys():
                ref_x, ref_y = self.card_buffer[key]['center_box']
                if abs(x - ref_x) < self.card_margin and abs(y - ref_y) < self.card_margin: # find card box in buffer of cards
                    buffer_key = key
                    break
            
            if buffer_key == -1: # card not in buffer
                l = ['0'] * 5 # init labels array with [0,0,0,0,0]
                l[0] = labels[i] # first elem is actual label
                self.card_buffer[self.buffer_counter] = {'center_box': (x, y), 'found_labels': l, 'counter': 1}
                self.buffer_counter += 1
            else: # found card, update center box and label array 
                card_b = self.card_buffer[buffer_key]
                l = card_b['found_labels']
                l[card_b['counter']] = labels[i] # add new found label
                c = (card_b['counter'] + 1) % 5 # add 1 to counter for found labels, but in range [0,5]
                self.card_buffer[buffer_key] = {'center_box': (x, y), 'found_labels': l, 'counter': c}

                occurence = Counter(l) # how often occures each label for this box
                for elem in occurence:
                    if occurence[elem] > 2 and not elem == '0': # if the same label was detected 3 or more times for this box
                        new_labels.append(elem) # append the max label
                        new_boxes.append(box) # and append the corresponding box
                        break
        return new_boxes, new_labels


    def update(self, boxes, labels, confis):
        if not labels:
            if self.new_round_counter > 2:
                self.reset_round()
            else:
                self.new_round_counter += 1
            return
        self.show_new_round = False
        boxes, labels = self.buffer_cards(boxes, labels, confis)
        new_player_cards = [[] for _ in range(self.num_players)] # create [[], [], [], []]
        new_player_boxes = [[] for _ in range(self.num_players)]
        for i in range(len(labels)):
            if labels[i] in self.available_cards:
                self.available_cards.remove(labels[i]) # card not available anymore
            player_num = self.get_player_to_box(boxes[i]) # this card belongs to which player?
            new_player_cards[player_num].append(labels[i])
            new_player_boxes[player_num].append(boxes[i])

        for i in range(len(self.players)):
            self.players[i].get_new_cards(new_player_boxes[i], new_player_cards[i]) # assign the card to the corresponding player
            self.players[i].predict_winning(self.available_cards) # calculate the win chances for each player
            

    def show_chance(self, image):
        if self.show_new_round:
            text_size, _ = cv2.getTextSize("New round", cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1.2, thickness=2)
            cv2.putText(image, "New round", (int(self.half_width-text_size[0]/2), self.half_height), cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1.2, color=(167, 231, 54), thickness=2)  # color
            return image

        for player in self.players:
            x, y = player.show_pos
            cv2.putText(image, "{}: {}".format(player.player_name, player.handvalue), (x, y), cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.8, color=(255, 255, 0), thickness=1)  # show playername and handvalue
            cv2.putText(image, "{}".format(player.handcards), (x, y+30), cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.8, color=(255, 255, 0), thickness=1)  # show player handcards
            win = player.win_chance
            on_point = player.right_chance
            if win > -1:
                cv2.putText(image, "Draw card: {}".format(win), (x, y+60), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6, color=(0, 255, 0), thickness=1)  # green color
            if on_point > -1:
                cv2.putText(image, "Win chance: {}".format(on_point), (x, y+90), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6, color=(100, 200, 0), thickness=1)  # red color
            if player.handvalue == 21:
                text_size, _ = cv2.getTextSize("{} won!!!".format(player.player_name), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1.2, thickness=2)
                cv2.putText(image, "{} won!!!".format(player.player_name), (int(self.half_width-text_size[0]/2), self.half_height), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1.2, color=(167, 231, 54), thickness=2)  # color
        return image
