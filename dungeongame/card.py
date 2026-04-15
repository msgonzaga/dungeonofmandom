from dungeongame.equipment import Equipment


class Card:
    def __init__(self, name: str, value: int, dies_to: list[Equipment]):
        self.name = name
        self.value = value
        self.dies_to = dies_to
    
    def __str__(self):
        return f"Name: {self.name}, Dies to: {[str(eqp) for eqp in self.dies_to]}, Value: {self.value}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)