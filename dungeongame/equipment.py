from enum import Enum


class Equipment(str, Enum):
    ARMOR = "[A]rmor",
    SHIELD = "[S]hield",
    CHALICE = "[C]halice",
    TORCH = "[T]orch",
    VORPAL_SWORD = "[V]orpal Sword",
    LANCE = "[L]ance"

    def __str__(self):
        return self.value.replace("[", "").replace("]", "")
