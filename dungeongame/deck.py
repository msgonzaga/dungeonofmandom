from dungeongame.card import Card
from random import shuffle


class Deck:
    def __init__(self, cards: list[Card] = []):
        self.cards = cards

    def add_card(self, card):
        self.cards.append(card)

    def remove_card(self, card):
        self.cards.remove(card)

    def draw(self):
        return self.cards.pop()

    def shuffle(self):
        shuffle(self.cards)

    def __str__(self):
        return str([card.name for card in self.cards])

    def __len__(self):
        return len(self.cards)
    
    def __bool__(self):
        return bool(self.cards)
