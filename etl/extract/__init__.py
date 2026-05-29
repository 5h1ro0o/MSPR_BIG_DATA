from .elections import extract_elections
from .candidats import extract_candidats_raw
from .demographique import extract_demographique
from .chomage import extract_chomage_historique, extract_emploi_2022
from .pauvrete import extract_pauvrete

__all__ = [
    "extract_elections",
    "extract_candidats_raw",
    "extract_demographique",
    "extract_chomage_historique",
    "extract_emploi_2022",
    "extract_pauvrete",
]
