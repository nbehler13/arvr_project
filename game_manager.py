from player import Player
import cv2
import numpy as np

class GameManager:

    def __init__(self, VID_WIDTH, VID_HEIGHT):
        self.num_players = 4
        self.players = []
        self.half_width = VID_WIDTH / 2
        self.half_height = VID_HEIGHT / 2

        for i in range(self.num_players):
            if i == 1:
                pos = (350, 60)
                name_pos = (400, 40)
            elif i == 2:
                pos = (350, 350)
                name_pos = (400, 280)
            elif i == 3:
                pos = (20, 350)
                name_pos = (80, 280)
            else:
                pos = (20, 60)
                name_pos = (80, 40)
            self.players.append(Player(i, pos, name_pos)) # instance of player 0, player 1, ..., player num_players

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


    def update(self, boxes, labels):
        for i in range(len(labels)):
            if labels[i] in self.available_cards:
                self.available_cards.remove(labels[i]) # card not available anymore
            player_num = self.get_player_to_box(boxes[i]) # this card belongs to which player?
            self.players[player_num].get_new_card(labels[i]) # assign the card to the corresponding player

        for player in self.players:
            player.predict_winning(self.available_cards) # calculate the win chances for each player
            

    def show_chance(self, image):
        for player in self.players:
            win = player.win_chance
            if win > -1:
                x, y = player.show_pos
                cv2.putText(image, str(win), (x, y), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=1)  # green color
                cv2.putText(image, str(100-win), (x, y+20), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=1)  # red color

            cv2.putText(image, "{}: {}".format(player.player_name, player.handvalue), player.name_pos, cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1, color=(255, 255, 0), thickness=2)  # red color
        return image
