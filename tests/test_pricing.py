"""Tests du moteur de tarification : cohérence métier et explicabilité."""

import pytest

from src.pricing import Profil, ProfilInvalide, compare, facteurs_principaux, simulate


def prime(**kwargs) -> float:
    return simulate(Profil(**kwargs)).prime_annuelle


# --- Cohérence métier -------------------------------------------------------


def test_jeune_conducteur_plus_cher_qu_experimente():
    assert prime(age=20, annees_permis=0) > prime(age=40, annees_permis=20)


def test_surprime_jeune_disparait_apres_deux_ans():
    # Règle réelle : 100 % → 50 % → 0 % (moitié du taux INITIAL par an)
    p0, p1, p2 = (prime(age=30, annees_permis=n) for n in (0, 1, 2))
    assert p0 > p1 > p2
    assert p2 == prime(age=30, annees_permis=5)


def test_conduite_accompagnee_reduit_la_surprime():
    assert prime(age=19, annees_permis=0, conduite_accompagnee=True) < prime(age=19, annees_permis=0)


def test_franchise_haute_baisse_la_prime_tous_risques():
    assert prime(franchise=1000) < prime(franchise=300) < prime(franchise=150)


def test_franchise_sans_effet_au_tiers():
    assert prime(formule="tiers", franchise=1000) == prime(formule="tiers", franchise=150)


def test_ordre_des_formules():
    assert prime(formule="tiers") < prime(formule="tiers_plus") < prime(formule="tous_risques")


def test_bonus_50_divise_la_prime_par_deux():
    assert prime(crm=0.5) == pytest.approx(prime(crm=1.0) / 2, abs=0.01)


def test_grande_ville_plus_chere_que_campagne():
    assert prime(zone="grande_ville") > prime(zone="urbaine") > prime(zone="rurale")


# --- Explicabilité ----------------------------------------------------------


def test_la_somme_des_impacts_explique_toute_la_prime():
    r = simulate(Profil(age=22, annees_permis=1, zone="grande_ville", crm=1.25))
    total = r.prime_base + sum(f.impact_euros for f in r.facteurs)
    assert total == pytest.approx(r.prime_annuelle, abs=0.05)


def test_chaque_facteur_cite_une_source_existante():
    from src.rag import KNOWLEDGE_DIR

    for f in simulate(Profil()).facteurs:
        assert (KNOWLEDGE_DIR / f.source).exists(), f.source


def test_facteur_principal_d_un_jeune_est_la_surprime():
    r = simulate(Profil(age=20, annees_permis=0))
    assert facteurs_principaux(r, 1)[0].nom == "Surprime jeune conducteur"


def test_compare_scenario():
    c = compare(Profil(), franchise=1000)
    assert c["delta"] < 0 and c["apres"].prime_annuelle < c["avant"].prime_annuelle


# --- Validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "profil",
    [
        Profil(age=20, annees_permis=10),  # permis avant 17 ans : impossible
        Profil(age=16, annees_permis=0),
        Profil(crm=0.3),  # sous le plancher légal de 0,50
        Profil(crm=4.0),  # au-dessus du plafond légal de 3,50
        Profil(formule="premium"),
        Profil(franchise=200),
        Profil(groupe_vehicule=12),
    ],
)
def test_profils_incoherents_rejetes(profil):
    with pytest.raises(ProfilInvalide):
        simulate(profil)