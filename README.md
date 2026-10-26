# Agent IA de simulation et d'explication de primes d'assurance auto

Agent **LangGraph** qui simule la prime annuelle d'une assurance auto, explique ce qui la compose, teste des scénarios (« et si… ») et répond aux questions sur l'assurance **en citant ses sources** (RAG).

> ⚠️ Projet de démonstration. Les coefficients de tarification sont **fictifs**. La valeur du projet est l'architecture : séparation calcul / rédaction, RAG auditable, garde-fou sur les montants et tests.

> 🔑 **Par défaut, l'agent utilise Claude et nécessite une clé API Anthropic** (voir [Installation](#6-installation)). Vous pouvez aussi le brancher sur un autre modèle, par exemple un modèle open source en local (voir [Changer de modèle](#8-changer-de-modèle-optionnel)).

---

## 1. Le problème

Un client qui reçoit un devis d'assurance auto voit un montant, mais ne comprend pas pourquoi il paie ce prix ni comment le faire baisser. Les conseillers passent beaucoup de temps à l'expliquer.

| Besoin du client | Exemple de question | Ce que fait l'agent |
|---|---|---|
| Obtenir un prix | « J'ai 24 ans, permis depuis 3 ans, tous risques, combien ? » | Appelle l'outil `simuler_prime` |
| Comprendre le prix | « Pourquoi c'est si cher ? » | Détaille l'impact en euros de chaque facteur |
| Tester un scénario | « Et si je prends une franchise de 1000 € ? » | Appelle `comparer_scenario` et montre l'écart |
| Comprendre l'assurance | « C'est quoi le bonus-malus ? » | Appelle `rechercher_documents` et cite la source |
| Hors sujet | « Quelle est la capitale de la France ? » | Refus poli, sans appeler d'outil |

**Contrainte métier centrale : un prix d'assurance ne doit jamais être inventé.** Le LLM rédige les explications, mais tous les montants viennent d'un moteur de calcul déterministe, et un garde-fou supprime tout montant qui n'en vient pas.

---

## 2. Glossaire

| Terme | Définition |
|---|---|
| **Prime** | Montant payé chaque année à l'assureur. |
| **Sinistre** | Accident ou dommage déclaré à l'assureur. |
| **Responsabilité civile (RC)** | Couverture des dégâts causés aux autres. Obligatoire. |
| **Tiers / Tiers + / Tous risques** | Les trois niveaux de couverture, du moins cher au plus complet. |
| **Franchise** | Part du sinistre payée par l'assuré. Franchise haute = prime basse. |
| **CRM (bonus-malus)** | Coefficient de 0,50 à 3,50 qui multiplie la prime selon l'historique d'accidents. |
| **Facteur de risque** | Caractéristique qui influence le prix (âge, véhicule, zone…). |
| **RAG** | Retrieval-Augmented Generation : le LLM répond à partir de documents retrouvés, pas de sa mémoire. |
| **Outil (tool)** | Fonction Python que le LLM peut appeler. Ici, le calcul et la recherche sont des outils. |

---

## 3. Comment la prime est calculée (modèle simplifié)

Modèle **multiplicatif**, comme dans les tarifs réels :

```
Prime = Prime de base (selon la formule)
        × coefficient âge / surprime jeune conducteur
        × coefficient véhicule
        × coefficient zone géographique
        × coefficient kilométrage
        × coefficient franchise
        × CRM (bonus-malus)
        + frais et taxes
```

Le moteur (`src/pricing.py`) renvoie non seulement le prix, mais aussi **le coefficient et l'impact en euros de chaque facteur**. C'est ce qui permet à l'agent d'expliquer le prix sans rien inventer.

---

## 4. Architecture

Un **agent unique** (Claude) qui raisonne et appelle des **outils déterministes**, suivi d'un **garde-fou** qui vérifie la réponse avant qu'elle parte au client.

```
                 ┌──────────────────────────┐
 Question ─────▶ │  Agent (Claude)          │ ◀──────────────┐
                 │  décide : répondre ou    │                │
                 │  appeler un outil        │                │
                 └────────────┬─────────────┘                │
                  appel d'outil│      │ réponse finale        │
                              ▼      │                       │
                 ┌──────────────────────────┐                │
                 │  Outils (déterministes)  │  résultat JSON │
                 │  • simuler_prime         │ ───────────────┘
                 │  • comparer_scenario     │
                 │  • rechercher_documents  │
                 └──────────────────────────┘
                                     │
                                     ▼
                 ┌──────────────────────────┐
                 │  Garde-fou               │
                 │  • montants vérifiés     │
                 │  • mention non contract. │
                 └────────────┬─────────────┘
                              ▼
                       Réponse au client
```

### Les briques

| Brique | Fichier | Rôle | LLM ? |
|---|---|---|---|
| **Agent** | `src/agent.py` | Lit la question, choisit l'outil, rédige la réponse. Suit des règles strictes (prompt système). | Oui |
| **simuler_prime** | `src/tools.py` → `pricing.simulate()` | Prime annuelle + impact de chaque facteur. | Non |
| **comparer_scenario** | `src/tools.py` → `pricing.compare()` | Prime avant / après une modification. | Non |
| **rechercher_documents** | `src/tools.py` → `rag.rechercher()` | 5 passages de la base de connaissances, avec leur source. | Non |
| **Garde-fou** | `src/guardrail.py` | Supprime tout montant absent des résultats d'outils ou des messages du client ; ajoute « estimation non contractuelle ». | Non |
| **Interface** | `app.py` | Conversation + panneau « ce que l'agent a fait ». Aucune logique métier. | Non |

### Points clés de conception

- **Schémas Pydantic** : les paramètres des outils sont typés (`Literal` pour la formule, la zone, la franchise). Une valeur invalide est rejetée avant l'appel et l'erreur revient au LLM pour qu'il se corrige.
- **Validation du profil** : un profil incohérent (ex. 20 ans et 10 ans de permis) renvoie une erreur que l'agent explique au client.
- **Mémoire de conversation** : un checkpointer LangGraph (`MemorySaver`) garde le profil d'un message à l'autre, rangé par session.
- **Limite d'étapes** : `recursion_limit = 12` empêche les boucles infinies agent ↔ outils.
- **Pannes** : si l'API ne répond pas (réseau, quota, clé invalide), l'agent renvoie un message d'indisponibilité au lieu de planter.

### Le RAG en bref

| Étape | Choix | Pourquoi |
|---|---|---|
| Chunking | Structurel : une section `##` = un chunk, redécoupée sans couper une phrase si elle dépasse 70 mots | Le modèle dense ne lit que 128 tokens |
| Embeddings | Dense `paraphrase-multilingual-MiniLM-L12-v2` + sparse BM25 (français) | Le dense comprend les paraphrases, le BM25 les mots exacts (« Hamon ») |
| Stockage | Qdrant en mode local (un dossier, pas de serveur) | Rien à installer |
| Recherche | Hybride, fusion RRF des deux classements | Meilleure que chacune seule (voir `python -m src.rag eval`) |
| Hors sujet | Rejet si le score dense **et** le score BM25 sont faibles | Évite de citer un passage sans rapport |

L'index n'est reconstruit que si les documents de `knowledge/` changent (empreinte du corpus).

### Le garde-fou en détail

1. Il extrait tous les nombres présents dans les **sources autorisées** : résultats d'outils, messages du client, franchises du catalogue.
2. Pour chaque montant de la réponse (nombre suivi de `€` ou `euros`), il vérifie qu'il existe dans ces sources (tolérance de 0,50 pour les arrondis). Il gère les formats français (`1 234,56 €`, `3.750 €`).
3. Un montant non justifié est remplacé par `[montant retiré : non issu du calcul]` et signalé dans l'interface.
4. Si un prix a été calculé dans le tour, il ajoute la mention « Estimation indicative et non contractuelle ».

---

## 5. Structure du dépôt

```
assurance-agent/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                     # Interface Streamlit
├── knowledge/                 # Base de connaissances (RAG), un fichier par thème
│   ├── bonus_malus.md
│   ├── garanties.md
│   ├── franchise.md
│   ├── facteurs_tarifaires.md
│   ├── jeunes_conducteurs.md
│   ├── reglementation.md
│   ├── marche_et_prix.md
│   └── statistiques_accidents.md
├── src/
│   ├── agent.py               # Graphe LangGraph : agent → outils → garde-fou
│   ├── tools.py               # Les 3 outils et leurs schémas Pydantic
│   ├── pricing.py             # Moteur de calcul de la prime
│   ├── rag.py                 # Chunking, embeddings, Qdrant, recherche hybride, évaluation
│   └── guardrail.py           # Vérification des montants + mention légale
├── tests/
│
│   # Créés automatiquement au premier lancement (non versionnés) :
├── .env                       # Votre clé API (à créer, voir 6.4)
├── qdrant_db/                 # Base vectorielle Qdrant en mode local
└── models/                    # Modèles d'embedding téléchargés
```

---

## 6. Installation

### Prérequis

- **Python 3.10+**
- **Une clé API Anthropic** (configuration par défaut). L'agent utilise Claude : sans clé, l'application ne démarre pas. Pour utiliser un autre modèle à la place, voir la section 8.
  1. Créer un compte sur la console Anthropic : https://console.anthropic.com/settings/keys
  2. Cliquer sur **Create Key** et copier la clé (elle ne sera plus affichée ensuite).
  3. Ajouter des crédits dans **Billing** : l'API est payante à l'usage. Le modèle par défaut (Claude Haiku) coûte quelques centimes pour une démo.

La clé se place dans un fichier `.env` (voir 6.4).

### 6.1 Récupérer le projet

```bash
git clone <url-du-depot>
cd <nom-du-dossier>
```

### 6.2 Créer et activer l'environnement virtuel

```bash
python -m venv .venv
```

Puis l'activer (à refaire **à chaque nouveau terminal**) :

| Système | Commande |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

Quand il est actif, `(.venv)` apparaît au début de la ligne du terminal. Pour en sortir : `deactivate`.

> Sous PowerShell, si l'activation est bloquée : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis relancer la commande.

### 6.3 Installer les dépendances

```bash
pip install -r requirements.txt
```

### 6.4 Configurer la clé API (fichier `.env`)

La clé n'est **pas** saisie avec `export` : elle est lue automatiquement depuis un fichier `.env`.

Créez un fichier nommé `.env` à la racine du projet (à côté de `app.py`) avec cette ligne :

```
ANTHROPIC_API_KEY=sk-ant-votre-cle
```


---

## 7. Utilisation

Toujours avec l'environnement virtuel **activé**, depuis la racine du projet.

### Interface web (Streamlit)

```bash
streamlit run app.py
```

L'interface s'ouvre dans le navigateur sur http://localhost:8501.

- **À gauche** : la conversation. Des exemples cliquables sont dans la barre latérale.
- **À droite** : ce que l'agent a fait : son chemin dans le graphe, les outils appelés, la décomposition de la prime (graphique), les sources du RAG et le contrôle du garde-fou.

Pour arrêter : `Ctrl + C` dans le terminal.

> ⏳ **Premier lancement plus long (connexion Internet nécessaire)** : les modèles d'embedding (≈ 220 Mo) sont téléchargés dans `models/`, puis la base `qdrant_db/` est construite à partir de `knowledge/`. Tout est automatique. Les lancements suivants sont rapides.

> ⚠️ Qdrant en mode local n'accepte qu'**un seul programme à la fois** sur `qdrant_db/`. Arrêtez Streamlit avant de lancer `python -m src.rag …`, et inversement.

### En ligne de commande

```bash
python -m src.agent          # conversation libre (q pour quitter)
python -m src.agent demo     # scénario de démonstration en 4 messages
```

### Le RAG seul

```bash
python -m src.rag index                  # (re)construit la base Qdrant
python -m src.rag chunks                 # statistiques du découpage
python -m src.rag "ma question"          # recherche hybride
python -m src.rag compare "ma question"  # dense vs sparse vs hybride
python -m src.rag eval                   # hit@3 et MRR sur 20 questions
```

### Tests

```bash
pytest -v
```

### Scénario de démonstration

```
Client : J'ai 24 ans, permis depuis 3 ans, bonus 0,90, voiture groupe 7, tous risques, en ville.
Agent  : Prime estimée : XXX € par an. Les facteurs qui pèsent le plus : …
Client : Pourquoi c'est si cher ?
Agent  : Le facteur principal est la surprime jeune conducteur (+XXX €) …
Client : Et si je prends une franchise de 1000 € ?
Agent  : Votre prime passerait à XXX € (− XX €). …
Client : C'est quoi le bonus-malus ?
Agent  : … [Source : bonus_malus.md › Principe du CRM]
```

---

## 8. Changer de modèle (optionnel)

Le modèle est libre : vous pouvez garder Claude, en choisir un autre, ou tester avec un modèle open source. Seule contrainte : **le modèle doit gérer l'appel d'outils (*tool calling*)**, sinon il ne pourra pas appeler le moteur de calcul ni le RAG.

### Autre modèle Claude

Ajoutez simplement la ligne suivante dans `.env` :

```
MODELE_LLM=claude-sonnet-5-5
```

### Modèle open source en local (Ollama)

Aucune clé API ni frais : le modèle tourne sur votre machine.

1. Installer [Ollama](https://ollama.com), puis télécharger un modèle compatible avec les outils :
   ```bash
   ollama pull qwen3
   ```
2. Installer le connecteur LangChain :
   ```bash
   pip install langchain-ollama
   ```
3. Dans `src/agent.py`, remplacer le contenu de la fonction `creer_llm()` :
   ```python
   def creer_llm():
       from langchain_ollama import ChatOllama

       return ChatOllama(model=os.environ.get("MODELE_LLM", "qwen3"), temperature=0)
   ```
4. Indiquer le modèle dans `.env` (la clé Anthropic n'est plus nécessaire) :
   ```
   MODELE_LLM=qwen3
   ```

Tout le reste fonctionne sans changement : outils, RAG, garde-fou et interface.

> 💡 Les petits modèles locaux suivent moins bien les règles du prompt système (ils inventent plus souvent un montant, ou oublient d'appeler un outil). C'est justement là que le **garde-fou** montre son utilité : tout montant inventé est retiré et signalé dans l'interface.

Le même principe s'applique à tout autre fournisseur compatible LangChain (OpenAI, Mistral…) : il suffit de remplacer le contenu de `creer_llm()`.

---

## 9. Points à défendre

- **Le LLM ne fixe jamais un prix.** Il appelle un outil déterministe et recopie le résultat. Le garde-fou le vérifie.
- **Explicabilité.** Chaque euro de la prime est rattaché à un facteur : c'est ce qu'attendent un client et un régulateur.
- **Gouvernance.** Réponses sourcées (RAG), mention non contractuelle, montants retirés tracés et affichés.
- **Robustesse.** Validation Pydantic des paramètres, erreurs d'outil renvoyées au LLM, limite d'étapes, message clair en cas de panne de l'API.

---

## 10. Limites et évolutions pour la production

| Aujourd'hui | En production |
|---|---|
| Coefficients fictifs | Modèle tarifaire réel (GLM ou gradient boosting) exposé en API |
| BM25 + dense sans re-ranking (CPU) | Ajout d'un re-ranker (cross-encoder) sur GPU |
| Qdrant en mode local (un seul programme à la fois) | Serveur Qdrant : `QdrantClient(url="http://...")` |
| Mémoire en RAM (`MemorySaver`) | Checkpointer persistant (PostgreSQL, Redis) |
| Tests sur un petit jeu de questions | Évaluation continue du RAG et des réponses |
| Traçabilité affichée dans l'interface | Journalisation de chaque décision (audit) |
| Données en clair | RGPD : masquage des données personnelles, hébergement UE |