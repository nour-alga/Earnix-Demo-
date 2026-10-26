"""
Ce fichier est fourni par claude 
Moteur de tarification de l'assurance auto.

Rôle : calculer une prime annuelle de façon DÉTERMINISTE et EXPLICABLE.
Le moteur calcul et on passe ca au llm qui explique 
Modèle multiplicatif (comme les tarifs réels issus d'un GLM) :
    prime = prime de base × coef_1 × coef_2 × ... × coef_n

⚠️ Les coefficients sont FICTIFS (calibrés pour donner des ordres de grandeur
réalistes, cf. knowledge/marche_et_prix.md). Seules deux règles sont réelles :
le bonus-malus (CRM) et le plafond légal de la surprime jeune conducteur.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

# --------------------------------------------------------------------------
# 1. Tables de coefficients (fictives sauf mention contraire)
# --------------------------------------------------------------------------

FORMULES = ("tiers", "tiers_plus", "tous_risques")
ZONES = ("rurale", "urbaine", "grande_ville")
FRANCHISES = (150, 300, 500, 1000)

PRIME_BASE = {"tiers": 520.0, "tiers_plus": 640.0, "tous_risques": 900.0}

COEF_ZONE = {"rurale": 0.85, "urbaine": 1.00, "grande_ville": 1.25}

# La franchise ne joue que sur les garanties dommages : au tiers, elle n'a presque pas d'effet.
COEF_FRANCHISE = {150: 1.10, 300: 1.00, 500: 0.93, 1000: 0.85}

FRAIS_ET_TAXES = 1.25  # frais de gestion, marge et taxes (fictif)

# Bornes réelles du bonus-malus (annexe à l'article A121-1 du code des assurances)
CRM_MIN, CRM_MAX = 0.50, 3.50


def coef_age(age: int) -> float:
    """Inspiré du risque par âge publié par l'ONISR (knowledge/statistiques_accidents.md)."""
    if age < 21:
        return 1.15
    if age < 25:
        return 1.08
    if age < 30:
        return 1.03
    if age < 65:
        return 1.00
    if age < 75:
        return 1.05
    return 1.15


def coef_surprime_jeune(annees_permis: int, conduite_accompagnee: bool) -> float:
    """Règle RÉELLE (article A121-1-1) appliquée au taux maximal.

    Plafond 100 % (50 % après conduite accompagnée), réduit de la moitié du taux
    INITIAL chaque année sans sinistre responsable : 100 % → 50 % → 0 %.
    On utilise l'ancienneté du permis comme approximation des années d'assurance.
    """
    taux_initial = 0.50 if conduite_accompagnee else 1.00
    taux = max(0.0, taux_initial - annees_permis * taux_initial / 2)
    return 1.0 + taux


def coef_vehicule(groupe: int) -> float:
    """Groupe 1 (petite citadine) à 10 (véhicule puissant ou cher) : de 0,80 à 1,43."""
    return round(0.80 + 0.07 * (groupe - 1), 2)


def coef_km(km_annuel: int) -> float:
    if km_annuel < 5_000:
        return 0.85
    if km_annuel < 10_000:
        return 0.95
    if km_annuel < 15_000:
        return 1.00
    if km_annuel < 20_000:
        return 1.08
    return 1.15


# --------------------------------------------------------------------------
# 2. Le profil du conducteur
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Profil:
    age: int = 35
    annees_permis: int = 15
    crm: float = 1.00  # coefficient bonus-malus
    groupe_vehicule: int = 5  # de 1 à 10
    zone: str = "urbaine"
    km_annuel: int = 12_000
    formule: str = "tous_risques"
    franchise: int = 300
    conduite_accompagnee: bool = False


def valider_profil(p: Profil) -> list[str]:
    """Renvoie la liste des incohérences (vide si le profil est valide)."""
    erreurs = []
    if not 18 <= p.age <= 100:
        erreurs.append("L'âge doit être compris entre 18 et 100 ans.")
    # En France, le permis B est accessible dès 17 ans (depuis 2024).
    if p.annees_permis < 0 or p.annees_permis > p.age - 17:
        erreurs.append(
            f"Ancienneté de permis incohérente : {p.annees_permis} ans de permis à {p.age} ans "
            "(le permis s'obtient au plus tôt à 17 ans)."
        )
    if not CRM_MIN <= p.crm <= CRM_MAX:
        erreurs.append(f"Le bonus-malus doit être compris entre {CRM_MIN} et {CRM_MAX}.")
    if not 1 <= p.groupe_vehicule <= 10:
        erreurs.append("Le groupe du véhicule doit être compris entre 1 et 10.")
    if p.zone not in ZONES:
        erreurs.append(f"Zone inconnue : {p.zone}. Valeurs possibles : {', '.join(ZONES)}.")
    if p.formule not in FORMULES:
        erreurs.append(f"Formule inconnue : {p.formule}. Valeurs possibles : {', '.join(FORMULES)}.")
    if p.franchise not in FRANCHISES:
        erreurs.append(f"Franchise non proposée : {p.franchise} €. Valeurs possibles : {FRANCHISES}.")
    if not 0 <= p.km_annuel <= 100_000:
        erreurs.append("Le kilométrage annuel doit être compris entre 0 et 100 000 km.")
    return erreurs


# --------------------------------------------------------------------------
# 3. Le résultat, explicable facteur par facteur
# --------------------------------------------------------------------------


@dataclass
class Facteur:
    nom: str
    valeur: str  # la donnée du client, lisible
    coefficient: float
    impact_euros: float  # combien ce facteur ajoute (+) ou retire (−) à la prime
    source: str  # fichier de la base de connaissances qui explique ce facteur


@dataclass
class Resultat:
    prime_annuelle: float
    prime_mensuelle: float
    prime_base: float
    facteurs: list[Facteur] = field(default_factory=list)
    mention: str = "Estimation indicative et non contractuelle, calculée avec des coefficients fictifs."

    def to_dict(self) -> dict:
        return asdict(self)


class ProfilInvalide(ValueError):
    def __init__(self, erreurs: list[str]):
        super().__init__(" ".join(erreurs))
        self.erreurs = erreurs


def simulate(p: Profil) -> Resultat:
    """Calcule la prime et l'impact en euros de chaque facteur.

    L'impact est calculé de façon séquentielle, dans un ordre fixe : on applique les
    coefficients un par un et on mesure de combien chacun fait bouger le montant.
    La somme des impacts + la prime de base = la prime finale (au centime près).
    """
    erreurs = valider_profil(p)
    if erreurs:
        raise ProfilInvalide(erreurs)

    franchise_coef = 1.0 if p.formule == "tiers" else COEF_FRANCHISE[p.franchise]
    surprime = coef_surprime_jeune(p.annees_permis, p.conduite_accompagnee)

    etapes = [
        ("Âge", f"{p.age} ans", coef_age(p.age), "statistiques_accidents.md"),
        (
            "Surprime jeune conducteur",
            f"{p.annees_permis} an(s) de permis" + (", conduite accompagnée" if p.conduite_accompagnee else ""),
            surprime,
            "jeunes_conducteurs.md",
        ),
        ("Véhicule", f"groupe {p.groupe_vehicule}", coef_vehicule(p.groupe_vehicule), "facteurs_tarifaires.md"),
        ("Zone", p.zone.replace("_", " "), COEF_ZONE[p.zone], "facteurs_tarifaires.md"),
        ("Kilométrage", f"{p.km_annuel:,} km/an".replace(",", " "), coef_km(p.km_annuel), "facteurs_tarifaires.md"),
        (
            "Franchise",
            f"{p.franchise} €" + (" (sans effet au tiers)" if p.formule == "tiers" else ""),
            franchise_coef,
            "franchise.md",
        ),
        ("Bonus-malus", f"coefficient {p.crm:.2f}", p.crm, "bonus_malus.md"),
        ("Frais de gestion et taxes", "forfait", FRAIS_ET_TAXES, "marche_et_prix.md"),
    ]

    base = PRIME_BASE[p.formule]
    montant = base
    facteurs = []
    for nom, valeur, coef, source in etapes:
        nouveau = montant * coef
        facteurs.append(Facteur(nom, valeur, round(coef, 4), round(nouveau - montant, 2), source))
        montant = nouveau

    prime = round(montant, 2)
    return Resultat(
        prime_annuelle=prime,
        prime_mensuelle=round(prime / 12, 2),
        prime_base=base,
        facteurs=facteurs,
    )


def compare(p: Profil, **modifications) -> dict:
    """Scénario « et si… » : recalcule la prime avec un ou plusieurs changements."""
    nouveau = replace(p, **modifications)
    avant, apres = simulate(p), simulate(nouveau)
    delta = round(apres.prime_annuelle - avant.prime_annuelle, 2)
    return {
        "modifications": modifications,
        "avant": avant,
        "apres": apres,
        "delta": delta,
        "delta_pct": round(100 * delta / avant.prime_annuelle, 1),
    }


def facteurs_principaux(r: Resultat, n: int = 3) -> list[Facteur]:
    """Les n facteurs qui pèsent le plus (en valeur absolue), hors frais et taxes."""
    candidats = [f for f in r.facteurs if f.nom != "Frais de gestion et taxes" and f.impact_euros != 0]
    return sorted(candidats, key=lambda f: abs(f.impact_euros), reverse=True)[:n]


def euros(x: float) -> str:
    """Format français : 1 234,56 €."""
    return f"{x:,.2f}".replace(",", " ").replace(".", ",") + " €"


if __name__ == "__main__":
    # Démo : python -m src.pricing
    jeune = Profil(age=20, annees_permis=1, groupe_vehicule=4, formule="tous_risques")
    r = simulate(jeune)
    print(f"Prime de base ({jeune.formule}) : {euros(r.prime_base)}")
    for f in r.facteurs:
        signe = "+" if f.impact_euros >= 0 else "−"
        print(f"  {f.nom:<28} {f.valeur:<32} × {f.coefficient:<6} {signe} {euros(abs(f.impact_euros))}")
    print(f"Prime annuelle : {euros(r.prime_annuelle)} ({euros(r.prime_mensuelle)} par mois)")
    c = compare(jeune, franchise=1000)
    print(f"\nEt si franchise 1000 € ? {euros(c['apres'].prime_annuelle)} ({c['delta_pct']} %)")