from colors import Colors
import pygame as pg

class Text(object):
    @staticmethod
    def createText(name, x, y):
        font = pg.font.Font('freesansbold.ttf', 15)
        text = font.render(name, True, Colors.WHITE, Colors.BACKGROUND)
        text_rect = text.get_rect()
        text_rect.center = (x, y)
        return text, text_rect