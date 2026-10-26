
from __future__ import annotations

import json
import os
import sys
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src import guardrail
from src.pricing import FRANCHISES
from src.tools import OUTILS
from dotenv import load_dotenv   
load_dotenv()                    


MODELE_LLM = os.environ.get("MODELE_LLM", "claude-haiku-5-5")
MAX_ETAPES = 12  # nombre maximal de passages dans le graphe pour eviter les boucles infinis 

OUTILS_PRIX = {"simuler_prime", "comparer_scenario"}

MESSAGE_INDISPONIBLE = (
    "Le service est momentanément indisponible.Reesayer dans un instant "
    
)

# Le prompt système 

PROMPT_SYSTEME = """Tu es un assistant d'assurance automobile. Tu aides un client à obtenir,
comprendre et faire baisser le prix de son assurance auto. Tu réponds en français, de façon
claire et concise.

Règles absolues :
1. Tu ne calcules, n'estimes et n'inventes JAMAIS un montant. Tout montant en euros vient
   d'un outil (simuler_prime ou comparer_scenario). Tu le recopies tel quel.
2. Pour simuler une prime, l'âge et l'ancienneté du permis sont obligatoires : s'ils manquent,
   demande-les avant d'appeler l'outil. Pour le reste, utilise les valeurs par défaut et dis au
   client quelles hypothèses tu as retenues.
3. Pour expliquer un prix (« pourquoi c'est si cher ? »), appuie-toi sur les facteurs du calcul
   et leur impact en euros, en commençant par ceux qui pèsent le plus.
4. Pour un scénario (« et si… »), appelle comparer_scenario avec le profil actuel COMPLET et
   seulement les modifications.
5. Pour une question de connaissance, appelle rechercher_documents et réponds UNIQUEMENT à partir
   des passages renvoyés, en citant la source sous la forme [Source : fichier › section].
   Si aucun passage ne répond, dis que tu n'as pas l'information.
6. Si un outil renvoie une erreur de profil, explique-la simplement et demande une correction.
7. Si la question ne concerne pas l'assurance auto, refuse poliment, sans appeler d'outil."""

# mise en place de l'état du graph 
# MessagesState contient la liste des messages (client, agent, outils).



class EtatAgent(MessagesState):
    mode: str  # "llm", ou "erreur" si l'API n'a pas répondu
    montants_retires: list[str]  # ce que le garde-fou a supprimé


def texte_de(message: BaseMessage) -> str:
    """Le contenu texte d'un message (Claude peut renvoyer une liste de blocs)."""
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if isinstance(b, dict))


def messages_du_tour(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Les messages depuis la dernière question du client."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def resultat_valide(contenu: str) -> bool:
    """Un résultat d'outil de prix sans erreur (utilisé par le garde-fou)."""
    try:
        return "erreur" not in json.loads(contenu)
    except json.JSONDecodeError:
        return False


#appele aux llm 


def creer_llm():
    """Claude, avec une clé API obligatoire."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY manquante : définis cette variable d'environnement avant de lancer l'agent."
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=MODELE_LLM, max_tokens=1024, timeout=30, max_retries=1)


#construction du grah 

def construire_agent(llm="auto", checkpointer=None):
    """Assemble le graphe. llm="auto" (ou None) : Claude ; un objet : llm."""
    if llm in ("auto", None):
        llm = creer_llm()
    llm_outils = llm.bind_tools(OUTILS)

    #  Noeud 1 : l'agent décide
    def agent(etat: EtatAgent) -> dict:
        try:
            reponse = llm_outils.invoke([SystemMessage(PROMPT_SYSTEME), *etat["messages"]])
            return {"messages": [reponse], "mode": "llm"}
        except Exception as e:  # réseau coupé, clé invalide, quota…
            print(f"[agent] LLM indisponible ({type(e).__name__}: {e}).", file=sys.stderr)
            return {"messages": [AIMessage(content=MESSAGE_INDISPONIBLE)], "mode": "erreur"}

    # Noeud 2 : les outils
    # handle_tool_errors : une erreur (paramètre invalide…) revient au LLM sous forme
    # de message, au lieu de faire planter le programme.
    outils = ToolNode(OUTILS, handle_tool_errors=True)

    # Noeud 3 : le garde-fou 
    def garde_fou(etat: EtatAgent) -> dict:
        messages = etat["messages"]
        finale = messages[-1]
        # Références autorisées : tout ce que les outils ont renvoyé et tout ce que
        # le client a écrit, depuis le début de la conversation.
        sources = [texte_de(m) for m in messages if isinstance(m, (ToolMessage, HumanMessage))]
        # Les valeurs du catalogue (franchises proposées) ne sont pas des prix inventés.
        # (une valeur par ligne : « 150 300 » serait lu comme cent cinquante mille trois cents)
        sources.append("\n".join(str(f) for f in FRANCHISES))
        prix_calcule = any(
            isinstance(m, ToolMessage) and m.name in OUTILS_PRIX and resultat_valide(texte_de(m))
            for m in messages_du_tour(messages)
        )
        texte, retires = guardrail.appliquer(texte_de(finale), sources, prix_calcule)
        # Même id = on REMPLACE le message de l'agent par sa version vérifiée.
        return {"messages": [AIMessage(content=texte, id=finale.id)], "montants_retires": retires}

   
    def apres_agent(etat: EtatAgent) -> str:
        return "outils" if getattr(etat["messages"][-1], "tool_calls", None) else "garde_fou"

    graphe = StateGraph(EtatAgent)
    graphe.add_node("agent", agent)
    graphe.add_node("outils", outils)
    graphe.add_node("garde_fou", garde_fou)
    graphe.add_edge(START, "agent")
    graphe.add_conditional_edges("agent", apres_agent, ["outils", "garde_fou"])
    graphe.add_edge("outils", "agent")
    graphe.add_edge("garde_fou", END)
    return graphe.compile(checkpointer=checkpointer or MemorySaver())


def repondre(agent, question: str, session: str = "demo") -> dict:
    """Pose une question à l'agent et renvoie tout ce qu'il faut pour l'afficher et l'auditer."""
    config = {"configurable": {"thread_id": session}, "recursion_limit": MAX_ETAPES}
    try:
        etat = agent.invoke({"messages": [HumanMessage(question)], "montants_retires": []}, config)
    except GraphRecursionError:
        return {
            "reponse": "Je n'ai pas réussi à traiter cette demande. Pouvez-vous la reformuler ?",
            "outils": [], "resultats": [], "montants_retires": [], "mode": "erreur",
        }
    tour = messages_du_tour(etat["messages"])
    return {
        "reponse": texte_de(etat["messages"][-1]),
        "outils": [(a["name"], a["args"]) for m in tour for a in (getattr(m, "tool_calls", None) or [])],
        "resultats": [(m.name, texte_de(m)) for m in tour if isinstance(m, ToolMessage)],
        "montants_retires": etat.get("montants_retires", []),
        "mode": etat.get("mode", "?"),
    }


# ============================================================================
#Sénario 
# ============================================================================

SCENARIO_DEMO = [
    "J'ai 24 ans, permis depuis 3 ans, bonus 0,90, voiture groupe 7, tous risques, en ville.",
    "Pourquoi c'est si cher ?",
    "Et si je prends une franchise de 1000 € ?",
    "C'est quoi le bonus-malus ?",
]

if __name__ == "__main__":
    try:
        agent = construire_agent()
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    session = uuid.uuid4().hex
    print(f"Mode : Claude ({MODELE_LLM})\n")

    questions = SCENARIO_DEMO if sys.argv[1:] == ["demo"] else None
    while True:
        if questions is not None:
            if not questions:
                break
            question = questions.pop(0)
            print(f"Client : {question}")
        else:
            question = input("Client : ").strip()
            if question.lower() in ("", "q", "quit", "exit"):
                break
        r = repondre(agent, question, session)
        for nom, args in r["outils"]:
            print(f"  ⚙ {nom}({json.dumps(args, ensure_ascii=False)})")
        if r["montants_retires"]:
            print(f"  🛡 garde-fou : montants retirés {r['montants_retires']}")
        print(f"Agent  : {r['reponse']}\n")