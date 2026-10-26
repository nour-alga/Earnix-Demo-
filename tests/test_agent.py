"""
Tests de l'agent.

  1. OUTILS        — appelés directement, sans LLM.
  2. GARDE-FOU     — un montant inventé est supprimé, un vrai est gardé.
  3. MÉMOIRE       — la conversation est renvoyée au LLM, les sessions sont isolées.
  4. FAUX LLM      — on SIMULE un LLM pour tester la boucle et les pannes :
                     LLM qui invente un prix, LLM en panne, LLM qui boucle.
  5. VRAI LLM      — ignoré sans ANTHROPIC_API_KEY.

La recherche RAG est remplacée par une fausse (les vrais tests du RAG sont dans
test_rag.py) : on teste ici l'AGENT, pas les embeddings.
python -m pytest tests/test_agent.py -v
"""

import json
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import agent as A
from src import guardrail, rag
from src.tools import comparer_scenario, rechercher_documents, simuler_prime

PROFIL_DEMO = {"age": 24, "annees_permis": 3, "crm": 0.9, "groupe_vehicule": 7, "formule": "tous_risques", "zone": "urbaine"}


@pytest.fixture(autouse=True)
def faux_rag(monkeypatch):
    def rechercher(question, k=3, mode="hybride"):
        if "tarte" in question:
            return []
        return [
            rag.Resultat(
                "bonus_malus.md › Principe du CRM", "bonus_malus.md", "Principe du CRM",
                "Le CRM multiplie la prime. Il commence à 1,00.", ["Code des assurances, article A121-1"], 1.0,
            )
        ]

    monkeypatch.setattr(rag, "rechercher", rechercher)


class FauxLLM:
    """Remplace Claude. `script` est une fonction : messages → réponse (ou exception)."""

    def __init__(self, script):
        self.script = script

    def bind_tools(self, outils):
        return self

    def invoke(self, messages):
        return self.script(messages)


# ============================================================================
# 1. OUTILS
# ============================================================================


def test_simuler_prime_renvoie_prime_et_facteurs():
    d = json.loads(simuler_prime.invoke({"profil": PROFIL_DEMO}))
    assert d["prime_annuelle"] > 0
    assert len(d["facteurs"]) == 8
    total = d["prime_base"] + sum(f["impact_euros"] for f in d["facteurs"])
    assert total == pytest.approx(d["prime_annuelle"], abs=0.05)  # chaque euro est expliqué


def test_simuler_prime_profil_incoherent_renvoie_une_erreur_lisible():
    d = json.loads(simuler_prime.invoke({"profil": {"age": 20, "annees_permis": 10}}))
    assert d["erreur"] == "profil incohérent"
    assert "permis" in d["details"][0]


def test_schema_refuse_une_valeur_hors_liste():
    with pytest.raises(Exception):  # Pydantic : franchise de 200 € non proposée
        simuler_prime.invoke({"profil": {"age": 30, "annees_permis": 5, "franchise": 200}})


def test_age_et_permis_obligatoires_dans_le_schema():
    schema = simuler_prime.args_schema.model_json_schema()
    profil = schema["$defs"]["ProfilArgs"]
    assert set(profil["required"]) == {"age", "annees_permis"}


def test_comparer_scenario_franchise_haute_baisse_la_prime():
    d = json.loads(comparer_scenario.invoke({"profil": PROFIL_DEMO, "modifications": {"franchise": 1000}}))
    assert d["difference_annuelle"] < 0
    assert d["apres"]["prime_annuelle"] < d["avant"]["prime_annuelle"]


def test_comparer_scenario_sans_modification():
    d = json.loads(comparer_scenario.invoke({"profil": PROFIL_DEMO, "modifications": {}}))
    assert "erreur" in d


def test_rechercher_documents_cite_ses_sources():
    d = json.loads(rechercher_documents.invoke({"question": "bonus malus"}))
    assert d["passages"][0]["citation"] == "bonus_malus.md › Principe du CRM"
    assert json.loads(rechercher_documents.invoke({"question": "tarte"}))["passages"] == []


# ============================================================================
# 2. GARDE-FOU
# ============================================================================

RESULTAT = '{"prime_annuelle": 1334.07, "prime_mensuelle": 111.17, "difference_annuelle": -200.11}'


def test_garde_fou_supprime_un_montant_invente():
    texte, retires = guardrail.appliquer("Votre prime est de 1 334,07 €, un concurrent fait 999 €.", [RESULTAT], True)
    assert "1 334,07 €" in texte
    assert "999 €" not in texte and retires == ["999 €"]
    assert guardrail.MONTANT_RETIRE in texte


def test_garde_fou_accepte_arrondi_et_valeur_absolue():
    texte, retires = guardrail.appliquer("Environ 1 334 € par an, soit 200 € d'économie.", [RESULTAT], True)
    assert retires == []


def test_garde_fou_accepte_les_formats_et_les_montants_du_client():
    sources = [RESULTAT, "Et si je prends une franchise de 1000 € ?"]
    texte, retires = guardrail.appliquer("1334.07 euros ; franchise 1 000 € ; 111,17€ par mois.", sources, True)
    assert retires == []


def test_mention_ajoutee_seulement_si_un_prix_a_ete_calcule():
    assert guardrail.MENTION in guardrail.appliquer("Votre prime : 1 334,07 €.", [RESULTAT], True)[0]
    assert guardrail.MENTION not in guardrail.appliquer("Le CRM multiplie la prime.", [], False)[0]


def test_conversion_des_nombres():
    assert guardrail.vers_float("1 960,03") == 1960.03
    assert guardrail.vers_float("1960.03") == 1960.03
    assert guardrail.vers_float("3.750") == 3750
    assert guardrail.vers_float("0,85") == 0.85


# ============================================================================
# 3. MÉMOIRE : ce que le LLM reçoit d'un message à l'autre
# ============================================================================


def _questions_vues(messages):
    """Les questions du client que le LLM a reçues dans cet appel."""
    return [m.content for m in messages if isinstance(m, HumanMessage)]


def test_la_memoire_renvoie_toute_la_conversation_au_llm():
    vus = []

    def script(messages):
        vus.append(_questions_vues(messages))
        return AIMessage(content="D'accord.")

    agent = A.construire_agent(llm=FauxLLM(script))
    A.repondre(agent, "J'ai 24 ans, permis depuis 3 ans.", "memo")
    A.repondre(agent, "Et si je prends une franchise de 1000 € ?", "memo")
    # Au 2e message, le LLM voit AUSSI le 1er : c'est la mémoire.
    assert vus[-1] == ["J'ai 24 ans, permis depuis 3 ans.", "Et si je prends une franchise de 1000 € ?"]


def test_le_profil_du_premier_calcul_est_reutilise_pour_le_scenario():
    def script(messages):
        dernier = messages[-1]
        if isinstance(dernier, ToolMessage):  # un outil vient de répondre : on rédige
            return AIMessage(content="Voici le résultat.")
        if "franchise" in dernier.content:  # 2e message : on retrouve le profil DANS L'HISTORIQUE
            profil = next(
                a["args"]["profil"] for m in messages for a in (getattr(m, "tool_calls", None) or [])
                if a["name"] == "simuler_prime"
            )
            appel = {"name": "comparer_scenario", "args": {"profil": profil, "modifications": {"franchise": 1000}}, "id": "c2"}
            return AIMessage(content="", tool_calls=[appel])
        return AIMessage(content="", tool_calls=[{"name": "simuler_prime", "args": {"profil": PROFIL_DEMO}, "id": "c1"}])

    agent = A.construire_agent(llm=FauxLLM(script))
    A.repondre(agent, A.SCENARIO_DEMO[0], "scenario")
    r = A.repondre(agent, A.SCENARIO_DEMO[2], "scenario")
    assert r["outils"][0][0] == "comparer_scenario"
    assert r["outils"][0][1]["profil"]["age"] == 24
    assert r["montants_retires"] == []


def test_deux_sessions_ne_partagent_pas_leur_memoire():
    vus = []

    def script(messages):
        vus.append(_questions_vues(messages))
        return AIMessage(content="D'accord.")

    agent = A.construire_agent(llm=FauxLLM(script))
    A.repondre(agent, "J'ai 24 ans.", "client_a")
    A.repondre(agent, "Et si je prends une franchise de 1000 € ?", "client_b")
    assert vus[-1] == ["Et si je prends une franchise de 1000 € ?"]  # le client B ne voit rien du client A


# ============================================================================
# 4. FAUX LLM : la boucle de l'agent et les pannes
# ============================================================================


def test_llm_qui_invente_un_prix_est_corrige_par_le_garde_fou():
    def script(messages):
        if not isinstance(messages[-1], ToolMessage):  # 1er tour : appeler l'outil
            return AIMessage(content="", tool_calls=[{"name": "simuler_prime", "args": {"profil": PROFIL_DEMO}, "id": "c1"}])
        # 2e tour : il recopie le bon prix… et invente une promotion
        return AIMessage(content="Votre prime est de 1 334,07 € par an. Avec notre promo, seulement 899 € !")

    r = A.repondre(A.construire_agent(llm=FauxLLM(script)), "J'ai 24 ans…", "llm1")
    assert r["mode"] == "llm"
    assert "1 334,07 €" in r["reponse"]
    assert "899 €" not in r["reponse"] and r["montants_retires"] == ["899 €"]
    assert guardrail.MENTION in r["reponse"]


def test_franchise_du_catalogue_proposee_en_exemple_n_est_pas_retiree():
    # Cas réel vu en démo avec Haiku : « voulez-vous tester une franchise de 500 € ? »
    def script(messages):
        if not isinstance(messages[-1], ToolMessage):
            return AIMessage(content="", tool_calls=[{"name": "simuler_prime", "args": {"profil": PROFIL_DEMO}, "id": "c1"}])
        return AIMessage(content="1 334,07 € par an. Voulez-vous tester une franchise de 500 € ou de 1 000 € ?")

    r = A.repondre(A.construire_agent(llm=FauxLLM(script)), "J'ai 24 ans…", "catalogue")
    assert r["montants_retires"] == []
    assert "500 €" in r["reponse"]


def test_llm_qui_repond_un_prix_sans_appeler_l_outil():
    script = lambda messages: AIMessage(content="Pour votre profil, comptez environ 1 200 € par an.")
    r = A.repondre(A.construire_agent(llm=FauxLLM(script)), "J'ai 24 ans, combien ?", "llm2")
    assert "1 200 €" not in r["reponse"]


def test_llm_en_panne_renvoie_un_message_d_excuse():
    def script(messages):
        raise ConnectionError("réseau coupé")

    r = A.repondre(A.construire_agent(llm=FauxLLM(script)), A.SCENARIO_DEMO[0], "panne")
    assert r["mode"] == "erreur"
    assert A.MESSAGE_INDISPONIBLE in r["reponse"]
    assert r["outils"] == []


def test_llm_qui_boucle_est_arrete():
    def script(messages):  # appelle l'outil indéfiniment
        return AIMessage(content="", tool_calls=[{"name": "rechercher_documents", "args": {"question": "x"}, "id": f"c{len(messages)}"}])

    r = A.repondre(A.construire_agent(llm=FauxLLM(script)), "boucle", "boucle")
    assert r["mode"] == "erreur" and "reformuler" in r["reponse"]


def test_erreur_de_parametre_renvoyee_au_llm_sans_planter():
    vus = []

    def script(messages):
        if not isinstance(messages[-1], ToolMessage):
            bad = {"profil": {"age": 30, "annees_permis": 5, "franchise": 200}}
            return AIMessage(content="", tool_calls=[{"name": "simuler_prime", "args": bad, "id": "c1"}])
        vus.append(messages[-1].content)
        return AIMessage(content="Cette franchise n'est pas proposée : choisissez 150, 300, 500 ou 1000 €.")

    r = A.repondre(A.construire_agent(llm=FauxLLM(script)), "franchise 200", "err")
    assert vus and "Error" in vus[0]  # l'erreur Pydantic est revenue au LLM
    assert "pas proposée" in r["reponse"]


# ============================================================================
# 5. VRAI LLM (ignorés sans clé API)
# ============================================================================

avec_cle = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="pas de ANTHROPIC_API_KEY")


@avec_cle
def test_vrai_llm_simule_avec_l_outil():
    r = A.repondre(A.construire_agent(), A.SCENARIO_DEMO[0], "vrai")
    assert r["mode"] == "llm"
    assert [n for n, _ in r["outils"]] == ["simuler_prime"]
    assert "1 334" in r["reponse"]
    assert r["montants_retires"] == []


@avec_cle
def test_vrai_llm_refuse_le_hors_sujet_sans_outil():
    r = A.repondre(A.construire_agent(), "Quelle est la capitale de la France ?", "hs")
    assert r["outils"] == []


@avec_cle
def test_vrai_llm_demande_age_et_permis_avant_de_simuler():
    r = A.repondre(A.construire_agent(), "Je voudrais une simulation", "slot")
    assert r["outils"] == []