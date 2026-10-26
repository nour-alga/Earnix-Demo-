"""
Les guardrails
Règle : tout montant en euros cité dans la réponse doit exister dans ce que les
outil ont renvoyé (calculs, passages de la base) ou dans ce que le client a dit.
Sinon, il est supprimé. Ensuite, on ajoute la mention « non contractuelle ».
"""

from __future__ import annotations

import re
# tracabilité 
MENTION = "Estimation indicative et non contractuelle, calculée avec des coefficients fictifs."
MONTANT_RETIRE = "[montant retiré : non issu du calcul]"


_NOMBRE = r"\d{1,3}(?:[   .]\d{3})+(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?"
# Un montant = un nombre suivi de « € » ou « euros »
_MONTANT = re.compile(rf"(?<![\d,.])({_NOMBRE})\s?(?:€|euros?\b)", re.IGNORECASE)
_NOMBRE_SEUL = re.compile(rf"(?<![\d])-?({_NOMBRE})")
#  pour l'arrondissement des valeurs 
TOLERANCE = 0.5


def vers_float(texte: str) -> float:
    t = re.sub(r"[   ]", "", texte)
    if "," in t:  # virgule pour montre les décimales et les points = milliers
        t = t.replace(".", "").replace(",", ".")
    elif re.search(r"\.\d{3}$", t) and t.count(".") >= 1 and not re.search(r"\.\d{1,2}$", t):
        t = t.replace(".", "")  # 3.750 = trois mille sept cent cinquante
    return float(t)


def nombres_autorises(sources: list[str]) -> set[float]:
    """Tous les nombres présents dans les textes de référence (résultats d'outils,
    messages du client). On prend la valeur absolue : une baisse de −143 € peut
    être écrite « 143 € de moins »."""
    autorises = set()
    for texte in sources:
        for m in _NOMBRE_SEUL.finditer(texte):
            try:
                autorises.add(abs(vers_float(m.group(1))))
            except ValueError:
                pass
    return autorises


def verifier_montants(reponse: str, sources: list[str]) -> tuple[str, list[str]]:
    """Remplace chaque montant non justifié. Renvoie (réponse corrigée, montants retirés)."""
    autorises = nombres_autorises(sources)
    retires = []

    def remplacer(m: re.Match) -> str:
        valeur = vers_float(m.group(1))
        if any(abs(valeur - a) <= TOLERANCE for a in autorises):
            return m.group(0)
        retires.append(m.group(0))
        return MONTANT_RETIRE

    return _MONTANT.sub(remplacer, reponse), retires


def appliquer(reponse: str, sources: list[str], prix_calcule: bool) -> tuple[str, list[str]]:
    """Le garde-fou complet : montants vérifiés + mention si un prix a été calculé."""
    reponse, retires = verifier_montants(reponse, sources)
    if prix_calcule and "non contractuelle" not in reponse.lower():
        reponse = f"{reponse.rstrip()}\n\n_{MENTION}_"
    return reponse, retires