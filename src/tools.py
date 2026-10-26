"""
Les OUTILS de l'agent
=====================

L'agent (le LLM) ne calcule rien et ne cherche rien lui-même : il APPELLE ces
trois outils, qui exécutent du code déterministe.

    simuler_prime        → pricing.simulate()   : la prime et l'impact de chaque facteur
    comparer_scenario    → pricing.compare()    : « et si je change ma franchise ? »
    rechercher_documents → rag.rechercher()     : passages de la base, avec leur source

Comment le LLM sait quoi envoyer ? Grâce au SCHÉMA (classe Pydantic ProfilArgs) :
LangChain le transforme en description JSON que le LLM lit. Les types `Literal`
limitent les valeurs possibles (pas de formule « premium » inventée), et `Field`
explique chaque paramètre. Si le LLM envoie quand même une valeur invalide, Pydantic
la rejette avant même l'appel, et l'erreur revient au LLM pour qu'il se corrige.

Chaque outil renvoie du JSON : c'est ce texte que le garde-fou relira pour
vérifier que tout montant cité dans la réponse vient bien d'un calcul.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src import pricing, rag

# ============================================================================
# 1. Les schémas : ce que le LLM a le droit d'envoyer
# ============================================================================


class ProfilArgs(BaseModel):
    """Le profil du conducteur. Seuls l'âge et l'ancienneté du permis sont obligatoires."""

    age: int = Field(description="Âge du conducteur en années (18 à 100).")
    annees_permis: int = Field(description="Nombre d'années depuis l'obtention du permis B.")
    crm: float = Field(1.0, description="Coefficient bonus-malus (CRM), de 0.50 à 3.50. 1.00 par défaut.")
    groupe_vehicule: int = Field(5, description="Groupe du véhicule, de 1 (petite citadine) à 10 (puissant).")
    zone: Literal["rurale", "urbaine", "grande_ville"] = Field(
        "urbaine", description="Lieu où dort le véhicule. « en ville » = urbaine, Paris/Lyon/Marseille = grande_ville."
    )
    km_annuel: int = Field(12_000, description="Kilomètres parcourus par an.")
    formule: Literal["tiers", "tiers_plus", "tous_risques"] = Field("tous_risques", description="Niveau de couverture.")
    franchise: Literal[150, 300, 500, 1000] = Field(300, description="Montant de la franchise en euros.")
    conduite_accompagnee: bool = Field(False, description="Permis obtenu après conduite accompagnée (AAC).")

    def vers_profil(self) -> pricing.Profil:
        return pricing.Profil(**self.model_dump())


class Modifications(BaseModel):
    """Les changements à tester. Ne renseigner QUE ce qui change."""

    crm: Optional[float] = None
    groupe_vehicule: Optional[int] = None
    zone: Optional[Literal["rurale", "urbaine", "grande_ville"]] = None
    km_annuel: Optional[int] = None
    formule: Optional[Literal["tiers", "tiers_plus", "tous_risques"]] = None
    franchise: Optional[Literal[150, 300, 500, 1000]] = None
    conduite_accompagnee: Optional[bool] = None


# ============================================================================
# 2. Mise en forme des résultats (JSON lisible par le LLM et le garde-fou)
# ============================================================================


def _resultat_vers_dict(r: pricing.Resultat) -> dict:
    return {
        "prime_annuelle": r.prime_annuelle,
        "prime_mensuelle": r.prime_mensuelle,
        "prime_base": r.prime_base,
        "facteurs": [asdict(f) for f in r.facteurs],
        "facteurs_principaux": [f.nom for f in pricing.facteurs_principaux(r)],
    }


def _json(donnees: dict) -> str:
    return json.dumps(donnees, ensure_ascii=False)


def _erreur_profil(e: pricing.ProfilInvalide) -> str:
    # On ne lève PAS d'exception : on renvoie l'erreur au LLM, qui l'expliquera au client.
    return _json({"erreur": "profil incohérent", "details": e.erreurs})


# ============================================================================
# 3. Les outils
# ============================================================================


@tool
def simuler_prime(profil: ProfilArgs) -> str:
    """Calcule la prime annuelle d'assurance auto d'un conducteur, et l'impact en euros
    de chaque facteur (âge, surprime jeune conducteur, véhicule, zone, kilométrage,
    franchise, bonus-malus, frais). À utiliser pour TOUT montant de prime, et pour
    expliquer pourquoi une prime est élevée."""
    try:
        r = pricing.simulate(profil.vers_profil())
    except pricing.ProfilInvalide as e:
        return _erreur_profil(e)
    return _json({"profil": profil.model_dump(), **_resultat_vers_dict(r), "mention": r.mention})


@tool
def comparer_scenario(profil: ProfilArgs, modifications: Modifications) -> str:
    """Compare la prime actuelle avec la prime après un ou plusieurs changements
    (« et si je prends une franchise de 1000 € ? », « et si je passe au tiers ? »).
    `profil` = le profil ACTUEL complet ; `modifications` = seulement ce qui change."""
    changements = modifications.model_dump(exclude_none=True)
    if not changements:
        return _json({"erreur": "aucune modification indiquée"})
    try:
        c = pricing.compare(profil.vers_profil(), **changements)
    except pricing.ProfilInvalide as e:
        return _erreur_profil(e)
    return _json(
        {
            "modifications": changements,
            "avant": _resultat_vers_dict(c["avant"]),
            "apres": _resultat_vers_dict(c["apres"]),
            "difference_annuelle": c["delta"],
            "difference_pct": c["delta_pct"],
        }
    )


@tool
def rechercher_documents(question: str) -> str:
    """Cherche dans la base de connaissances sur l'assurance auto (bonus-malus,
    garanties, franchise, jeunes conducteurs, réglementation, facteurs de prix,
    marché, accidents). Renvoie des passages avec leur source à citer.
    À utiliser pour toute question de connaissance, au lieu de répondre de mémoire."""

    # 5 passages : le LLM lit et synthétise, un peu plus de contexte l'aide à trouver
    # la bonne information (le mode de secours, lui, n'affiche que le premier).
    resultats = rag.rechercher(question, k=5)
    if not resultats:
        return _json({"passages": [], "info": "Aucun passage pertinent dans la base de connaissances."})
    return _json(
        {"passages": [{"citation": r.citation, "texte": r.texte, "references": r.references} for r in resultats]}
    )


OUTILS = [simuler_prime, comparer_scenario, rechercher_documents]