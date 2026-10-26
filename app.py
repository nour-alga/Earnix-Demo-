#Fichier fourni pas claude 
"""
Interface de démonstration (Streamlit)
=====================================

    streamlit run app.py

À gauche : la conversation avec l'agent.
À droite : ce que l'agent a fait pour répondre (outils appelés, calcul détaillé,
sources du RAG, contrôle du garde-fou). On voit le système « réfléchir ».

L'interface ne contient AUCUNE logique métier : elle appelle seulement
`repondre()` de src/agent.py et affiche ce qu'il renvoie.
"""

from __future__ import annotations

import json
import uuid

import altair as alt
import pandas as pd
import streamlit as st

from src import rag
from src.agent import MODELE_LLM, SCENARIO_DEMO, construire_agent, creer_llm, repondre
from src.pricing import euros

st.set_page_config(page_title="Agent assurance auto", page_icon="🚗", layout="wide")

# Couleurs du graphique (palette validée pour le daltonisme) : hausse / baisse de prime
ROUGE_HAUSSE, BLEU_BAISSE = "#e34948", "#2a78d6"

EXEMPLES = SCENARIO_DEMO + [
    "J'ai 20 ans et 10 ans de permis",
    "Quelle est la capitale de la France ?",
]


# ============================================================================
# 1. Chargement (une seule fois, partagé entre les visiteurs)
# ============================================================================


@st.cache_resource(show_spinner="Chargement de la base de connaissances…")
def charger_agent():
    rag.indexer()  # prépare la base Qdrant et les modèles d'embedding au démarrage
    return construire_agent()


@st.cache_resource
def mode_llm() -> str:
    return f"Claude ({MODELE_LLM})" if creer_llm() else "Mode de secours (sans clé API)"


def nouvelle_conversation():
    st.session_state.session = uuid.uuid4().hex  # la mémoire de l'agent est rangée par session
    st.session_state.messages = []  # [{"role", "contenu", "details"}]


if "session" not in st.session_state:
    nouvelle_conversation()


# ============================================================================
# 2. Panneau de droite : ce que l'agent a fait
# ============================================================================


def graphique_facteurs(facteurs: list[dict]):
    """Barres horizontales : de combien chaque facteur fait monter ou baisser la prime."""
    df = pd.DataFrame(
        [
            {
                "Facteur": f["nom"],
                "Valeur": f["valeur"],
                "Impact (€)": f["impact_euros"],
                "Effet": "Hausse" if f["impact_euros"] > 0 else "Baisse",
                "Libellé": ("+" if f["impact_euros"] > 0 else "−") + euros(abs(f["impact_euros"])),
            }
            for f in facteurs
            if f["impact_euros"] != 0
        ]
    ).sort_values("Impact (€)", key=abs, ascending=False)

    base = alt.Chart(df).encode(
        y=alt.Y("Facteur:N", sort=None, title=None, axis=alt.Axis(labelLimit=180, domain=False, ticks=False)),
        x=alt.X("Impact (€):Q", title="Impact sur la prime (€)", axis=alt.Axis(grid=True, gridOpacity=0.3)),
    )
    barres = base.mark_bar(cornerRadiusEnd=4, height=16).encode(
        color=alt.Color(
            "Effet:N",
            scale=alt.Scale(domain=["Hausse", "Baisse"], range=[ROUGE_HAUSSE, BLEU_BAISSE]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["Facteur", "Valeur", alt.Tooltip("Libellé", title="Impact")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888", strokeWidth=1).encode(x="x:Q")
    # Une ligne de 34 px par facteur : toutes les étiquettes restent lisibles
    return (zero + barres).properties(height=alt.Step(34))


def afficher_calcul(d: dict):
    c1, c2 = st.columns(2)
    c1.metric("Prime annuelle", euros(d["prime_annuelle"]))
    c2.metric("Par mois", euros(d["prime_mensuelle"]))
    st.caption(f"Prime de base : {euros(d['prime_base'])} — chaque barre montre l'effet d'un facteur.")
    st.altair_chart(graphique_facteurs(d["facteurs"]), width="stretch")
    with st.expander("Tableau du calcul"):
        st.dataframe(
            pd.DataFrame(d["facteurs"]).rename(
                columns={"nom": "Facteur", "valeur": "Valeur", "coefficient": "Coefficient",
                         "impact_euros": "Impact (€)", "source": "Document explicatif"}
            ),
            hide_index=True,
            width="stretch",
        )


def afficher_details(details: dict):
    libelles = {"llm": f"Claude ({MODELE_LLM})", "hors_ligne": "mode de secours (règles)",
                "hors_ligne (secours)": "mode de secours (le LLM n'a pas répondu)", "erreur": "erreur"}
    st.caption(f"Moteur : **{libelles.get(details['mode'], details['mode'])}**")

    # Le chemin suivi dans le graphe
    etapes = ["agent"] + [f"⚙ {nom}" for nom, _ in details["outils"]] + (["agent"] if details["outils"] else []) + ["🛡 garde-fou"]
    st.markdown("**Chemin dans le graphe :** " + " → ".join(f"`{e}`" for e in etapes))

    # Le garde-fou
    if details["montants_retires"]:
        st.error("🛡 Garde-fou : montants retirés car absents des calculs : " + ", ".join(details["montants_retires"]))
    elif any(nom in ("simuler_prime", "comparer_scenario") for nom, _ in details["resultats"]):
        st.success("🛡 Garde-fou : tous les montants cités viennent du moteur de calcul.")

    # Les résultats des outils
    for nom, contenu in details["resultats"]:
        try:
            d = json.loads(contenu)
        except json.JSONDecodeError:
            st.warning(f"{nom} : {contenu}")
            continue

        if "erreur" in d:
            st.warning("Profil refusé par la validation : " + " ".join(d.get("details", [d["erreur"]])))
        elif nom == "simuler_prime":
            st.subheader("Calcul de la prime")
            afficher_calcul(d)
        elif nom == "comparer_scenario":
            st.subheader("Scénario « et si… »")
            c1, c2 = st.columns(2)
            c1.metric("Avant", euros(d["avant"]["prime_annuelle"]))
            c2.metric("Après", euros(d["apres"]["prime_annuelle"]),
                      delta=f"{d['difference_annuelle']:+.2f} €".replace(".", ","), delta_color="inverse")
            st.caption("Modifications : " + ", ".join(f"{k} = {v}" for k, v in d["modifications"].items()))
            st.altair_chart(graphique_facteurs(d["apres"]["facteurs"]), width="stretch")
        elif nom == "rechercher_documents":
            st.subheader("Sources consultées (RAG)")
            if not d["passages"]:
                st.info("Aucun passage pertinent dans la base de connaissances.")
            for p in d["passages"]:
                with st.expander(f"📄 {p['citation']}"):
                    st.write(p["texte"])
                    if p["references"]:
                        st.caption("Références : " + " · ".join(p["references"]))


# ============================================================================
# 3. Barre latérale
# ============================================================================

with st.sidebar:
    st.title("🚗 Agent assurance auto")
    st.caption(mode_llm())
    st.markdown(
        "Simule une prime, explique ce qui la compose, teste des scénarios et répond "
        "aux questions sur l'assurance en citant ses sources."
    )
    st.markdown("**Essayez :**")
    for i, exemple in enumerate(EXEMPLES):
        if st.button(exemple, key=f"ex{i}", width="stretch"):
            st.session_state.question_en_attente = exemple
    st.divider()
    st.button("🔄 Nouvelle conversation", on_click=nouvelle_conversation, width="stretch")
    st.caption(
        "Architecture : un agent LangGraph, 3 outils déterministes (calcul, scénario, RAG hybride Qdrant) "
        "et un garde-fou qui vérifie chaque montant. Coefficients fictifs : estimation non contractuelle."
    )


# ============================================================================
# 4. Page principale
# ============================================================================

agent = charger_agent()
gauche, droite = st.columns([3, 2], gap="large")

question = st.chat_input("Votre question (ex. : J'ai 30 ans, 10 ans de permis, tous risques à Lyon)")
question = question or st.session_state.pop("question_en_attente", None)

if question:
    st.session_state.messages.append({"role": "user", "contenu": question})
    with gauche, st.spinner("L'agent réfléchit…"):
        r = repondre(agent, question, st.session_state.session)
    st.session_state.messages.append({"role": "assistant", "contenu": r["reponse"], "details": r})

with gauche:
    st.subheader("Conversation")
    if not st.session_state.messages:
        st.info("Posez une question, ou cliquez sur un exemple à gauche.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "🤖"):
            st.markdown(m["contenu"])

with droite:
    st.subheader("Ce que l'agent a fait")
    derniers = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if derniers:
        afficher_details(derniers[-1]["details"])
    else:
        st.caption("Les outils appelés, le calcul détaillé, les sources et le contrôle du garde-fou s'afficheront ici.")