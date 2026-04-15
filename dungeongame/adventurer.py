from dungeongame.equipment import Equipment
from dungeongame.card import Card


class Adventurer:
    def __init__(self):
        self.hp: int = 11
        self.equipment: list[Equipment] = [
            Equipment.ARMOR,
            Equipment.SHIELD,
            Equipment.CHALICE,
            Equipment.TORCH,
            Equipment.VORPAL_SWORD,
            Equipment.LANCE,
        ]
        self.vorpal_target: Card = None

    def remove_equipment(self, equipment: Equipment):
        if equipment in self.equipment:
            self.equipment.remove(equipment)
            if equipment == Equipment.ARMOR:
                self.hp -= 5
            elif equipment == Equipment.SHIELD:
                self.hp -= 3

    def take_damage(self, damage: int):
        self.hp -= damage
    
    def __str__(self):
        return f"HP: {self.hp}, Equipment: {[equipment.name for equipment in self.equipment]}"
