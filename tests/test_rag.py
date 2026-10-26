"""
Tests du pipeline RAG.

Trois familles :
  1. CHUNKING      — sur de petits documents écrits pour piéger le découpage.
  2. QDRANT        — indexation, réindexation, recherche dense / sparse / hybride,
                     garde-fou hors sujet. Les modèles sont remplacés par de faux
                     embeddings (rapides, sans téléchargement) : on teste NOTRE code.
  3. VRAIS MODÈLES — sur ta vraie base knowledge/. Ignorés si les modèles ne sont
                     pas téléchargés (lance d'abord : python -m src.rag index).
"""

import hashlib
import math
import re
import unicodedata

import pytest
from qdrant_client import models

from src import rag

# ============================================================================
# Documents de test
# ============================================================================

DOC_BONUS = """# Le bonus-malus

Ce document explique le coefficient de réduction-majoration.

## Principe du CRM

Le CRM est un coefficient qui multiplie la prime. Il commence à 1,00.

## Calcul du malus

Chaque accident responsable augmente le coefficient de 25 %.

## Calcul du malus

Un accident partiellement responsable augmente le coefficient de 12,5 %.

## Section vide

## Sources

- Code des assurances, article A121-1
* Service-public.fr
"""

DOC_LOI = """# Réglementation

## Loi Hamon

La loi Hamon permet de résilier son contrat après un an, à tout moment, sans frais.

## Obligation d'assurance

Rouler sans assurance est un délit puni d'une amende pouvant atteindre 3 750 euros.

## Sources

- Loi n° 2014-344
"""

DOC_VEHICULE = """# Garanties

## Vol du véhicule

La garantie vol couvre la disparition du véhicule après effraction.

## Bris de glace

Le pare-brise et les vitres sont couverts par la garantie bris de glace.

## Sources

- France Assureurs
"""


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Un petit dossier knowledge/ et une base Qdrant vides, propres à chaque test."""
    dossier = tmp_path / "knowledge"
    dossier.mkdir()
    (dossier / "bonus_malus.md").write_text(DOC_BONUS, encoding="utf-8")
    (dossier / "reglementation.md").write_text(DOC_LOI, encoding="utf-8")
    (dossier / "garanties.md").write_text(DOC_VEHICULE, encoding="utf-8")
    monkeypatch.setattr(rag, "KNOWLEDGE_DIR", dossier)
    monkeypatch.setattr(rag, "QDRANT_DIR", tmp_path / "qdrant_db")
    monkeypatch.setattr(rag, "_client", None)
    yield dossier
    if rag._client is not None:
        rag._client.close()
    rag._client = None


# ============================================================================
# 1. CHUNKING
# ============================================================================


def test_sections_lues_correctement(corpus):
    titre, sections, refs = rag.lire_sections(corpus / "bonus_malus.md")
    assert titre == "Le bonus-malus"
    noms = [s for s, _ in sections]
    assert noms == ["Introduction", "Principe du CRM", "Calcul du malus", "Calcul du malus"]
    assert "Section vide" not in noms  # pas de chunk vide
    assert refs == ["Code des assurances, article A121-1", "Service-public.fr"]  # « - » et « * »


def test_le_texte_avant_la_premiere_section_n_est_pas_perdu(corpus):
    _, sections, _ = rag.lire_sections(corpus / "bonus_malus.md")
    assert "coefficient de réduction-majoration" in sections[0][1]


def test_deux_sections_de_meme_titre_ont_des_ids_differents(corpus):
    chunks = rag.charger_chunks()
    assert len({c.id for c in chunks}) == len(chunks)


def test_ids_stables_entre_deux_chargements(corpus):
    assert [c.id for c in rag.charger_chunks()] == [c.id for c in rag.charger_chunks()]


def test_section_sources_non_indexee_mais_rattachee(corpus):
    chunks = rag.charger_chunks()
    assert all(c.section.lower() != "sources" for c in chunks)
    assert all(c.references for c in chunks)


def test_en_tete_de_contexte(corpus):
    c = rag.charger_chunks()[0]
    assert c.texte_a_indexer.startswith(f"{c.titre_doc} — {c.section}\n")


def test_dossier_vide_leve_une_erreur_claire(tmp_path):
    with pytest.raises(FileNotFoundError):
        rag.charger_chunks(tmp_path)


def test_tableau_long_coupe_avec_intro_et_entete_repetees():
    tableau = "Ce tableau liste des valeurs.\n\n| A | B |\n|---|---|\n" + "\n".join(
        f"| ligne {i} avec du texte | valeur {i} |" for i in range(30)
    )
    morceaux = rag.decouper_section(tableau)
    assert len(morceaux) > 1
    for m in morceaux:
        assert m.startswith("Ce tableau liste des valeurs.")
        assert "| A | B |" in m
        assert rag.nb_mots(m) <= rag.MAX_MOTS
    # Aucune ligne perdue ni dupliquée
    lignes = [l for m in morceaux for l in m.splitlines() if l.startswith("| ligne")]
    assert len(lignes) == 30


def test_une_phrase_n_est_jamais_coupee():
    phrases = [f"Ceci est la phrase numéro {i} de ce long paragraphe de test." for i in range(20)]
    morceaux = rag.decouper_section(" ".join(phrases))
    assert len(morceaux) > 1
    assert all(m.endswith(".") for m in morceaux)
    assert " ".join(morceaux) == " ".join(phrases)  # rien perdu


def test_une_longue_liste_est_coupee_entre_deux_elements():
    liste = "\n".join(f"- élément {i} de la liste sans point final" for i in range(25))
    morceaux = rag.decouper_section(liste)
    assert len(morceaux) > 1
    for m in morceaux:
        assert rag.nb_mots(m) <= rag.MAX_MOTS
        assert all(l.startswith("- élément") for l in m.splitlines())  # éléments entiers


def test_section_courte_gardee_entiere():
    assert rag.decouper_section("Une seule phrase courte.") == ["Une seule phrase courte."]


# ============================================================================
# 2. QDRANT — avec de faux embeddings
# ============================================================================
#
# Faux modèles volontairement simples : un mot = une dimension (via un hachage).
#   - faux dense  : sac de mots normalisé → le cosinus mesure les mots en commun
#   - faux sparse : un poids par mot (comme BM25)
# Ils suffisent à vérifier le stockage, l'IDF, la fusion RRF et le seuil.

MOTS_VIDES = set("le la les de des du un une et en a au aux est que qui quoi pour ce c il je mon ma si".split())


def _mots(texte):
    texte = unicodedata.normalize("NFKD", texte.lower()).encode("ascii", "ignore").decode()
    return [m[:6] for m in re.findall(r"[a-z0-9]+", texte) if m not in MOTS_VIDES and len(m) > 1]


def _h(mot, n):
    return int(hashlib.md5(mot.encode()).hexdigest(), 16) % n


def faux_dense(texte, dim=64):
    v = [0.0] * dim
    for m in _mots(texte):
        v[_h(m, dim)] += 1
    norme = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norme for x in v]


def faux_sparse(texte):
    poids = {}
    for m in _mots(texte):
        poids[_h(m, 100_000)] = poids.get(_h(m, 100_000), 0) + 1.0
    return models.SparseVector(indices=list(poids), values=list(poids.values()))


@pytest.fixture
def faux_modeles(monkeypatch):
    monkeypatch.setattr(rag, "embed_dense", lambda textes: [faux_dense(t) for t in textes])
    monkeypatch.setattr(rag, "embed_sparse", lambda textes: [faux_sparse(t) for t in textes])
    monkeypatch.setattr(rag, "embed_question_dense", faux_dense)
    monkeypatch.setattr(rag, "embed_question_sparse", faux_sparse)
    monkeypatch.setattr(rag, "SEUIL_PERTINENCE", 0.2)


def test_indexation_tous_les_chunks_dans_qdrant(corpus, faux_modeles):
    n = rag.indexer(forcer=True)
    assert n == len(rag.charger_chunks())
    point = rag.client_qdrant().scroll(rag.COLLECTION, limit=1, with_payload=True)[0][0]
    assert {"source", "section", "texte", "citation", "references"} <= set(point.payload)


def test_pas_de_reindexation_si_rien_n_a_change(corpus, faux_modeles, monkeypatch):
    rag.indexer()
    appels = []
    monkeypatch.setattr(rag, "embed_dense", lambda t: appels.append(1) or [faux_dense(x) for x in t])
    rag.indexer()
    assert appels == []  # les embeddings n'ont pas été recalculés


def test_reindexation_si_un_document_change(corpus, faux_modeles):
    n1 = rag.indexer()
    (corpus / "franchise.md").write_text("# Franchise\n\n## Franchise absolue\n\nMontant fixe.\n", encoding="utf-8")
    assert rag.indexer() == n1 + 1
    assert rag.rechercher("franchise absolue montant fixe")[0].source == "franchise.md"


def test_reindexation_ne_laisse_pas_de_chunk_obsolete(corpus, faux_modeles):
    rag.indexer()
    (corpus / "garanties.md").unlink()
    rag.indexer()
    sources = {p.payload["source"] for p in rag.client_qdrant().scroll(rag.COLLECTION, limit=100)[0]}
    assert "garanties.md" not in sources


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybride"])
def test_les_trois_modes_trouvent_la_bonne_section(corpus, faux_modeles, mode):
    r = rag.rechercher("comment résilier avec la loi Hamon", k=3, mode=mode)
    assert r, f"aucun résultat en mode {mode}"
    assert r[0].citation == "reglementation.md › Loi Hamon"
    assert r[0].references == ["Loi n° 2014-344"]


def test_hybride_renvoie_au_plus_k_resultats_sans_doublon(corpus, faux_modeles):
    r = rag.rechercher("accident coefficient malus", k=3)
    assert 0 < len(r) <= 3
    assert len({(x.citation, x.texte) for x in r}) == len(r)


def test_question_hors_sujet_ne_renvoie_rien(corpus, faux_modeles):
    assert rag.rechercher("recette tarte pommes cannelle") == []


def test_question_vide_ne_renvoie_rien(corpus, faux_modeles):
    assert rag.rechercher("   ") == []


def test_question_sans_mot_utile_ne_plante_pas(corpus, faux_modeles, monkeypatch):
    # Vecteur sparse vide (que des mots vides) → repli automatique sur le dense
    monkeypatch.setattr(rag, "embed_question_sparse", lambda q: models.SparseVector(indices=[], values=[]))
    assert rag.rechercher("vol du véhicule", mode="hybride")[0].section == "Vol du véhicule"


def test_mode_inconnu(corpus, faux_modeles):
    with pytest.raises(ValueError):
        rag.rechercher("bonus", mode="magique")


def test_verification_des_questions_d_evaluation(corpus):
    questions = [("q ok", "reglementation.md", "hamon"), ("q cassée", "reglementation.md", "inexistant")]
    assert rag.verifier_questions_eval(questions) == ["q cassée"]


def test_evaluation_calcule_hit_et_mrr(corpus, faux_modeles):
    questions = [("loi Hamon résilier", "reglementation.md", "hamon"), ("pare-brise vitres", "garanties.md", "bris")]
    scores = rag.evaluer(questions=questions)
    assert set(scores) == {"dense", "sparse", "hybride"}
    assert scores["hybride"] == {"hit@k": 1.0, "MRR": 1.0}


# ============================================================================
# 3. VRAIS MODÈLES, VRAIE BASE (ignorés s'ils ne sont pas disponibles)
# ============================================================================


def _vrais_modeles_disponibles():
    if not list(rag.KNOWLEDGE_DIR.glob("*.md")):
        return False
    try:
        rag.embed_question_dense("test")
        rag.embed_question_sparse("test")
        return True
    except Exception:
        return False


vrais_modeles = pytest.mark.skipif(
    not _vrais_modeles_disponibles(), reason="modèles non téléchargés : lance `python -m src.rag index`"
)


@pytest.fixture
def vraie_base():
    yield
    if rag._client is not None:
        rag._client.close()
        rag._client = None


@vrais_modeles
def test_vraie_base_chunks_valides():
    chunks = rag.charger_chunks()
    assert len({c.id for c in chunks}) == len(chunks)
    trop_longs = [c.citation for c in chunks if rag.nb_mots(c.texte) > rag.MAX_MOTS]
    assert trop_longs == [], f"chunks trop longs : {trop_longs}"


@vrais_modeles
def test_vraie_base_questions_d_evaluation_valides():
    assert rag.verifier_questions_eval() == []


@vrais_modeles
def test_vraie_base_hybride_hit3_au_moins_80(vraie_base):
    scores = rag.evaluer()
    assert scores["hybride"]["hit@k"] >= 0.8, scores


@vrais_modeles
def test_vraie_base_hors_sujet_refuse(vraie_base):
    # Le RAG rejette une question sans aucun rapport avec l'assurance.
    # Les cas ambigus (« capitale de la France ») sont refusés par le Routeur.
    assert rag.rechercher("donne-moi une recette de tarte aux pommes") == []


@vrais_modeles
def test_vraie_base_vraies_questions_jamais_rejetees(vraie_base):
    rejetees = [q for q, _, _ in rag.QUESTIONS_EVAL if not rag.rechercher(q)]
    assert rejetees == [], f"vraies questions rejetées comme hors sujet : {rejetees}"