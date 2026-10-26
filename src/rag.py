"""
les meuilleur pratique pour le RAG c'est la rechercher hybride (dense + sparse) et le reranking 
Dans ce fichier on utilise la recherche hybrid mais pas de reranking à cause de la CPU 

La pipeline RAG, dans l'ordre :

PIPELINE D'INDEXATION (une fois, quand les documents changent)
Étape 1 — Chunking   : découper les fichiers .md en morceaux (méthode utilisé :chunking structurelle) 
Étape 2 — Embeddings : transformer chaque chunk en 2 vecteurs
                        - dense  : modèle multilingue
                        - sparse : lBM25
Étape 3 : stockage des vecteurs dans quadrant  ( ici en local )

PIPELINE DE RECHERCHE (à chaque question):
Étape 4 — Recherche  :  HYBRIDE (fusion RRF des deux)

Utilisation (depuis la racine du projet) :
    python -m src.rag index                 → construit la base Qdrant
    python -m src.rag chunks                → montre le résultat du chunking
    python -m src.rag "ma question"         → cherche les passages pertinents (hybride)
    python -m src.rag compare "ma question" → dense vs sparse vs hybride, côte à côte
    python -m src.rag eval                  → compare les 3 modes sur 20 questions ( les  tests enfaite )
"""

from __future__ import annotations

import atexit
import hashlib
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient, models


RACINE = Path(__file__).resolve().parent.parent  # le dossier du projet
KNOWLEDGE_DIR = RACINE / "knowledge"  # les documents sources (.md)
QDRANT_DIR = RACINE / "qdrant_db"  # la base Qdrant (un simple dossier, pas de serveur)
MODELS_DIR = RACINE / "models"  # cache des modèles téléchargés
# Cration de collection qdrant 
COLLECTION = "assurance_auto" 

# Modèle DENSE : multilingue je l'ai choisi parceque c'est entrennait sur beaucoup de documents francais et en plus il est  léger (≈ 220 Mo), 384 dimensions.
# Mais à savoir qu'il ne lit que 128 tokens (≈ 80 mots). Au-delà, il ignore la fin.
MODELE_DENSE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELE_SPARSE = "Qdrant/bm25"

# Taille maximale d'un chunk, choisie pour rester sous la limite du modèle dense
MAX_MOTS = 70 # taille du chunk est dicté par le modèle ici 

# Nombre de candidats que chaque méthode propose avant la fusion hybride.
NB_CANDIDATS = 20

# Filet de sécurité « hors sujet ». Une question est rejetée seulement si les DEUX
SEUIL_PERTINENCE = 0.35  # cosinus dense du meilleur passage
SEUIL_SPARSE = 3.0  # score BM25 du meilleur passage



# ÉTAPE 1 :CHUNKING ( onutilise structured based chunking)
#   - une section « ## » = une idée = un chunk ;
#   - une section trop longue est redécoupée par blocs (paragraphe, liste, tableau),
#     sans jamais couper une phrase ni un élément de liste ;
#   - un tableau trop long est coupé par lignes, en répétant son introduction
#     et sa ligne d'en-tête dans chaque morceau ;
#   - chaque chunk reçoit un en-tête de contexte (titre du document + section).

@dataclass
class Chunk:
    """Un morceau de document, avec tout ce qu'il faut pour le citer."""

    source: str  # document de knoledge 
    titre_doc: str  # titre 
    section: str  # titre de la section 
    texte: str  # contenu du chunk
    references: list[str] = field(default_factory=list)  # sources officielles du fichier
    partie: int = 1  # numéro du morceau quand la section est coupé 
    nb_parties: int = 1
    num_section: int = 1  # position de la section dans le fichier

    @property
    def id(self) -> str:
        """Identifiant STABLE : le même chunk a toujours le même id.
        Si on réindexe, Qdrant remplace le chunk au lieu de créer un doublon.
        num_section évite une collision si deux sections portent le même titre
        (sinon la 2e écraserait la 1re dans Qdrant, sans aucune erreur)."""
        cle = f"{self.source}|{self.num_section}|{self.section}|{self.partie}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, cle))

    @property
    def texte_a_indexer(self) -> str:
        """enrichissement du text """
        return f"{self.titre_doc} — {self.section}\n{self.texte}"

    @property
    def citation(self) -> str:
        "citer source dans la reponse"
        suffixe = f" ({self.partie}/{self.nb_parties})" if self.nb_parties > 1 else ""
        return f"{self.source} › {self.section}{suffixe}"


def nb_mots(texte: str) -> int:
    return len(texte.split())


def lire_sections(fichier: Path) -> tuple[str, list[tuple[str, str]], list[str]]:
    """Lit un fichier .md et renvoie (titre du document, [(titre de section, contenu)], références)."""
    contenu = fichier.read_text(encoding="utf-8").lstrip("﻿")  # BOM éventuel (Windows)
    titre_doc = next((l[2:].strip() for l in contenu.splitlines() if l.startswith("# ")), fichier.stem)

    # On coupe le texte à chaque ligne qui commence par « ## ».
    blocs = re.split(r"^## +", contenu, flags=re.MULTILINE)

    sections, references = [], []
    # Le texte AVANT le premier « ## » (après le titre « # ») n'est pas perdu :
    # s'il existe, il devient une section « Introduction ».
    intro = "\n".join(l for l in blocs[0].splitlines() if not l.startswith("# ")).strip()
    if intro:
        sections.append(("Introduction", intro))

    for bloc in blocs[1:]:
        titre, _, corps = bloc.partition("\n")  # 1re ligne = titre, le reste = contenu
        titre, corps = titre.strip(), corps.strip()
        if titre.lower() == "sources":
            # La section Sources n'est pas indexée (elle n'aide pas à répondre),
            # mais elle est rattachée à chaque chunk pour la traçabilité.
            references = [re.sub(r"^[-*+]\s*", "", l.strip()) for l in corps.splitlines() if l.strip()]
        elif corps:  # une section vide ne produit pas de chunk
            sections.append((titre, corps))
    return titre_doc, sections, references


def est_tableau(bloc: str) -> bool:
    return bloc.lstrip().startswith("|")


def decouper_en_blocs(corps: str) -> list[str]:
    """Sépare une section en blocs (paragraphe, liste, tableau) aux lignes vides,
    puis colle à chaque tableau la phrase qui le présente (« Ce tableau montre… »)."""
    blocs = [b.strip() for b in re.split(r"\n\s*\n", corps) if b.strip()]
    resultat = []
    for b in blocs:
        if est_tableau(b) and resultat and not est_tableau(resultat[-1]):
            # Seule la DERNIÈRE phrase du paragraphe précédent présente le tableau.
            phrases = re.split(r"(?<=[.!?:])\s+", resultat.pop())
            if len(phrases) > 1:
                resultat.append(" ".join(phrases[:-1]))
            resultat.append(phrases[-1] + "\n\n" + b)
        else:
            resultat.append(b)
    return resultat


def couper_tableau(bloc: str) -> list[str]:
    """Coupe un tableau trop long par groupes de lignes. Chaque morceau garde
    l'introduction et la ligne d'en-tête : il reste compréhensible tout seul."""
    lignes = bloc.splitlines()
    debut = next(i for i, l in enumerate(lignes) if l.lstrip().startswith("|"))
    intro = "\n".join(lignes[:debut]).strip()
    entete, donnees = lignes[debut : debut + 2], lignes[debut + 2 :]  # en-tête + ligne |---|
    contexte = ([intro, ""] if intro else []) + entete

    morceaux, courant = [], []
    for ligne in donnees:
        if courant and nb_mots("\n".join(contexte + courant + [ligne])) > MAX_MOTS:
            morceaux.append("\n".join(contexte + courant))
            courant = []
        courant.append(ligne)
    morceaux.append("\n".join(contexte + courant))
    return morceaux


# Un élément de liste commence par « - », « * », « + » ou « 1. » / « 1) ».
_DEBUT_ITEM = r"\n(?=\s*(?:[-*+]|\d+[.)])\s)"


def couper_paragraphe(paragraphe: str) -> list[str]:
    """Coupe un paragraphe (ou une liste) trop long ENTRE deux phrases ou deux
    éléments de liste. Une phrase seule trop longue est gardée entière : la couper
    casserait son sens."""
    # Unités insécables : chaque élément de liste, puis chaque phrase à l'intérieur.
    unites = []  # (texte, séparateur à mettre AVANT)
    for i, item in enumerate(re.split(_DEBUT_ITEM, paragraphe)):
        for j, phrase in enumerate(re.split(r"(?<=[.!?])\s+", item.strip())):
            if phrase:
                unites.append((phrase, "\n" if (i > 0 and j == 0) else " "))

    morceaux, courant = [], ""
    for phrase, sep in unites:
        if courant and nb_mots(courant + " " + phrase) > MAX_MOTS:
            morceaux.append(courant)
            courant = ""
        courant = f"{courant}{sep}{phrase}".strip() if courant else phrase
    if courant:
        morceaux.append(courant)
    return morceaux


def decouper_section(corps: str) -> list[str]:
    """Renvoie la section entière si elle est assez courte, sinon des morceaux de MAX_MOTS au plus."""
    if nb_mots(corps) <= MAX_MOTS:
        return [corps]  # cas le plus fréquent

    # 1. Redécouper les blocs trop longs
    petits_blocs = []
    for b in decouper_en_blocs(corps):
        if nb_mots(b) <= MAX_MOTS:
            petits_blocs.append(b)
        elif "\n|" in b or est_tableau(b):
            petits_blocs.extend(couper_tableau(b))
        else:
            petits_blocs.extend(couper_paragraphe(b))

    # 2. Regrouper les petits blocs voisins tant qu'on reste sous la limite
    morceaux, courant = [], ""
    for b in petits_blocs:
        if courant and nb_mots(courant + "\n\n" + b) > MAX_MOTS:
            morceaux.append(courant)
            courant = ""
        courant = f"{courant}\n\n{b}".strip()
    morceaux.append(courant)
    return morceaux


def charger_chunks(dossier: Path | None = None) -> list[Chunk]:
    """ tous les fichiers .md → liste de chunks."""
    dossier = Path(dossier or KNOWLEDGE_DIR)
    chunks = []
    for fichier in sorted(dossier.glob("*.md")):
        titre_doc, sections, references = lire_sections(fichier)
        for num, (section, corps) in enumerate(sections, start=1):
            morceaux = decouper_section(corps)
            for i, texte in enumerate(morceaux, start=1):
                chunks.append(Chunk(fichier.name, titre_doc, section, texte, references, i, len(morceaux), num))
    if not chunks:
        raise FileNotFoundError(f"Aucun document .md trouvé dans {dossier}")
    return chunks



# EMBEDDINGS : transformer le texte en vecteurs


_modele_dense = None
_modele_sparse = None


def modele_dense():
    global _modele_dense
    if _modele_dense is None:
        from fastembed import TextEmbedding  

        _modele_dense = TextEmbedding(MODELE_DENSE, cache_dir=str(MODELS_DIR))
    return _modele_dense


def modele_sparse():
    global _modele_sparse
    if _modele_sparse is None:
        from fastembed import SparseTextEmbedding

        _modele_sparse = SparseTextEmbedding(MODELE_SPARSE, cache_dir=str(MODELS_DIR), language="french", avg_len=50)
    return _modele_sparse


def embed_dense(textes: list[str]) -> list[list[float]]:
    return [v.tolist() for v in modele_dense().embed(textes)]


def embed_sparse(textes: list[str]) -> list[models.SparseVector]:
    return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in modele_sparse().embed(textes)]


def embed_question_dense(question: str) -> list[float]:
    return next(iter(modele_dense().query_embed(question))).tolist()


def embed_question_sparse(question: str) -> models.SparseVector:
    e = next(iter(modele_sparse().query_embed(question)))
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())



# QDRANT : stocker les vecteurs dans la base vectorielle


# Qdrant tourne ici en MODE LOCAL : la base est un dossier (qdrant_db/), sans serveur.
# En production, on changerait juste une ligne : QdrantClient(url="http://...").

_client = None


def client_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        try:
            _client = QdrantClient(path=str(QDRANT_DIR))
        except RuntimeError as e:
            # Le mode local n'accepte qu'UN programme à la fois sur le dossier.
            raise RuntimeError(
                f"La base {QDRANT_DIR} est déjà ouverte par un autre programme "
                
            ) from e
        atexit.register(_client.close)  # fermer proprement la base à la sortie
    return _client


def empreinte_corpus(chunks: list[Chunk]) -> str:
    """Une « empreinte » du contenu : si un document change (texte, titre, nom
    de fichier, découpage), l'empreinte change et on sait qu'il faut réindexer."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.id}|{c.citation}|{c.texte_a_indexer}|{c.references}".encode("utf-8"))
    h.update(f"{MODELE_DENSE}|{MODELE_SPARSE}|{MAX_MOTS}".encode())
    return h.hexdigest()[:16]


def indexer(forcer: bool = False) -> int:
    """Pipeline d'indexation complet : chunking → embeddings → Qdrant.
    Ne refait rien si les documents n'ont pas changé (sauf si forcer=True)."""
    chunks = charger_chunks()
    empreinte = empreinte_corpus(chunks)
    fichier_empreinte = Path(QDRANT_DIR) / "empreinte.txt"
    client = client_qdrant()

    deja_a_jour = (
        not forcer
        and client.collection_exists(COLLECTION)
        and fichier_empreinte.exists()
        and fichier_empreinte.read_text() == empreinte
    )
    if deja_a_jour:
        return client.count(COLLECTION).count

    print(f"Indexation de {len(chunks)} chunks…")
    textes = [c.texte_a_indexer for c in chunks]
    vecteurs_denses = embed_dense(textes)
    vecteurs_sparses = embed_sparse(textes)

    # On recrée la collection à neuf 
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={
            # COSINE : on compare la DIRECTION des vecteurs (le sens), pas leur longueur.
            "dense": models.VectorParams(size=len(vecteurs_denses[0]), distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            # IDF : Qdrant calcule lui-même la rareté de chaque mot dans la collection.
            # Un mot rare (« Hamon ») pèse plus qu'un mot fréquent (« assurance »).
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )

    points = [
        models.PointStruct(
            id=c.id,
            vector={"dense": d, "sparse": s},
            payload={
                "source": c.source,
                "titre_doc": c.titre_doc,
                "section": c.section,
                "texte": c.texte,
                "citation": c.citation,
                "references": c.references,
            },
        )
        for c, d, s in zip(chunks, vecteurs_denses, vecteurs_sparses)
    ]
    client.upsert(COLLECTION, points)

    # Vérification : autant de points dans Qdrant que de chunks (sinon, collision d'id).
    nb = client.count(COLLECTION).count
    if nb != len(chunks):
        raise RuntimeError(f"{len(chunks)} chunks mais {nb} points dans Qdrant : des ids sont en double.")
    fichier_empreinte.write_text(empreinte)  # écrit APRÈS le succès seulement
    return nb



# RECHERCHE : dense, sparse ou hybride

#
# Recherche HYBRIDE = on lance les deux recherches, puis on FUSIONNE les classements
# avec RRF (Reciprocal Rank Fusion) 

@dataclass
class Resultat:
    citation: str
    source: str
    section: str
    texte: str
    references: list[str]
    score: float


def _vers_resultats(points) -> list[Resultat]:
    return [
        Resultat(
            citation=p.payload["citation"],
            source=p.payload["source"],
            section=p.payload["section"],
            texte=p.payload["texte"],
            references=p.payload["references"],
            score=round(p.score, 4),
        )
        for p in points
    ]


def scores_max(q_dense, q_sparse) -> tuple[float, float]:
    """Meilleur score dense (cosinus) et meilleur score sparse (BM25) de la question."""
    client = client_qdrant()
    d = client.query_points(COLLECTION, query=q_dense, using="dense", limit=1).points
    s = client.query_points(COLLECTION, query=q_sparse, using="sparse", limit=1).points if q_sparse.indices else []
    return (d[0].score if d else 0.0), (s[0].score if s else 0.0)


def rechercher(question: str, k: int = 3, mode: str = "hybride") -> list[Resultat]:
    """Renvoie les k passages les plus pertinents. mode = "dense", "sparse" ou "hybride"."""
    if mode not in ("dense", "sparse", "hybride"):
        raise ValueError(f"Mode inconnu : {mode} (dense, sparse ou hybride)")
    if not question or not question.strip():
        return []

    indexer()  # s'assure que la base existe et est à jour (rapide si rien n'a changé)
    client = client_qdrant()
    q_dense = embed_question_dense(question)
    q_sparse = embed_question_sparse(question)

    # Filet « hors sujet » : rejet seulement si le sens ET les mots sont loin du corpus.
    dense_max, sparse_max = scores_max(q_dense, q_sparse)
    if dense_max < SEUIL_PERTINENCE and sparse_max < SEUIL_SPARSE:
        return []

    # Une question faite uniquement de mots vides (« et si ? ») donne un vecteur
    # sparse VIDE : dans ce cas, seule la recherche dense a du sens.
    if mode != "dense" and not q_sparse.indices:
        mode = "dense"

    if mode == "dense":
        points = client.query_points(COLLECTION, query=q_dense, using="dense", limit=k, with_payload=True).points
    elif mode == "sparse":
        points = client.query_points(COLLECTION, query=q_sparse, using="sparse", limit=k, with_payload=True).points
    else:
        points = client.query_points(
            COLLECTION,
            prefetch=[  # 1. chaque méthode propose ses NB_CANDIDATS meilleurs candidats
                models.Prefetch(query=q_dense, using="dense", limit=NB_CANDIDATS),
                models.Prefetch(query=q_sparse, using="sparse", limit=NB_CANDIDATS),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),  # 2. fusion des deux classements
            limit=k,
            with_payload=True,
        ).points
    return _vers_resultats(points)


# ============================================================================
# ÉTAPE 5 — ÉVALUATION : mesurer au lieu de supposer
# ============================================================================
#
# Pour chaque question « dorée », on sait quelle section doit être trouvée.
#   - hit@3 : la bonne section est-elle dans les 3 premiers résultats ?
#   - MRR   : Mean Reciprocal Rank. 1 si la bonne section est 1re, 0,5 si 2e,
#             0,33 si 3e, 0 si absente. Récompense le fait d'être bien classé.
# Les questions sont volontairement variées : certaines reprennent les mots exacts
# des documents (favorables au sparse), d'autres sont des paraphrases (favorables au dense).

QUESTIONS_EVAL = [
    # (question, fichier attendu, mot présent dans le titre de la section attendue)
    ("c'est quoi le bonus malus", "bonus_malus.md", "principe"),
    ("comment est calculé le malus après un accident", "bonus_malus.md", "malus"),
    ("est-ce que je garde mon bonus si je change d'assureur", "bonus_malus.md", "assureur"),
    ("après combien d'années sans accident j'ai le bonus maximum", "bonus_malus.md", "bonus maximal"),
    ("combien de temps dure la surprime jeune conducteur", "jeunes_conducteurs.md", "durée"),
    ("la conduite accompagnée fait-elle baisser la prime", "jeunes_conducteurs.md", "accompagnée"),
    ("je viens d'avoir mon permis, pourquoi c'est si cher", "jeunes_conducteurs.md", "surprime"),
    ("différence entre tiers et tous risques", "garanties.md", "différence"),
    ("qu'est-ce que la responsabilité civile", "garanties.md", "responsabilité"),
    ("mon pare-brise est fissuré, suis-je couvert", "garanties.md", "bris de glace"),
    ("ma voiture a été volée", "garanties.md", "vol"),
    ("franchise absolue ou relative", "franchise.md", "franchise"),
    ("franchise en cas de catastrophe naturelle", "franchise.md", "catastrophe"),
    ("comment résilier mon assurance avec la loi Hamon", "reglementation.md", "hamon"),
    ("que risque-t-on à rouler sans assurance", "reglementation.md", "obligation"),
    ("c'est quoi la prime pure", "marche_et_prix.md", "prime pure"),
    ("quel est le prix moyen d'une assurance au tiers", "marche_et_prix.md", "prix"),
    ("quelles sont les causes des accidents mortels", "statistiques_accidents.md", "causes"),
    ("les assureurs peuvent-ils tarifer selon le sexe", "facteurs_tarifaires.md", "interdits"),
    ("pourquoi l'assurance coûte plus cher en ville", "facteurs_tarifaires.md", "zone"),
]

QUESTIONS_HORS_SUJET = [
    "quelle est la capitale de la France",
    "donne-moi une recette de tarte aux pommes",
    "qui a gagné la coupe du monde de football",
]


def verifier_questions_eval(questions=QUESTIONS_EVAL) -> list[str]:
    """Signale les questions dont la section attendue N'EXISTE PAS dans le corpus
    (fichier renommé, titre modifié…). Sans ce contrôle, une faute de frappe
    ferait baisser le score sans qu'on comprenne pourquoi."""
    sections = {(c.source, c.section.lower()) for c in charger_chunks()}
    return [q for q, f, mot in questions if not any(s == f and mot in t for s, t in sections)]


def evaluer(k: int = 3, questions=QUESTIONS_EVAL) -> dict[str, dict[str, float]]:
    """Calcule hit@k et MRR pour chaque mode de recherche."""
    scores = {}
    for mode in ("dense", "sparse", "hybride"):
        hits, rr = 0, 0.0
        for question, fichier, mot in questions:
            for rang, r in enumerate(rechercher(question, k=k, mode=mode), start=1):
                if r.source == fichier and mot in r.section.lower():
                    hits += 1 # pourcentage question réussi 
                    rr += 1 / rang #Mean Reciprocal Rank si c'est bien classe o pas 
                    break
        n = len(questions)
        scores[mode] = {"hit@k": round(hits / n, 3), "MRR": round(rr / n, 3)}
    return scores


# ============================================================================
# POINT D'ENTRÉE EN LIGNE DE COMMANDE
# ============================================================================


def _afficher_chunks() -> None:
    chunks = charger_chunks()
    tailles = [nb_mots(c.texte) for c in chunks]
    fichiers = sorted({c.source for c in chunks})
    print(f"{len(chunks)} chunks issus de {len(fichiers)} fichiers")
    print(f"Taille (mots) : min {min(tailles)}, moyenne {sum(tailles) / len(tailles):.0f}, max {max(tailles)}")
    print(f"Sections coupées en plusieurs morceaux : {len({(c.source, c.num_section) for c in chunks if c.nb_parties > 1})}")
    for f in fichiers:
        print(f"  {f:<28} {sum(c.source == f for c in chunks)} chunks")
    titres = {}
    for c in chunks:
        titres.setdefault((c.source, c.section), set()).add(c.num_section)
    doublons = [f"{s} › {t}" for (s, t), nums in titres.items() if len(nums) > 1]
    if doublons:
        print(f"⚠ Titres de section en double (citations ambiguës) : {', '.join(doublons)}")
    exemple = next((c for c in chunks if c.nb_parties > 1), chunks[0])
    print(f"\nExemple — {exemple.citation} (id {exemple.id[:8]}…)\n{'-' * 60}\n{exemple.texte_a_indexer}\n{'-' * 60}")


def _afficher(resultats: list[Resultat]) -> None:
    if not resultats:
        print("  Aucun passage pertinent (question hors sujet ?).")
    for rang, r in enumerate(resultats, start=1):
        print(f"  {rang}. [{r.score}] {r.citation}\n     {r.texte[:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    args = sys.argv[1:]
    commande = args[0] if args else ""

    if commande == "index":
        n = indexer(forcer=True)
        print(f"{n} chunks indexés dans Qdrant ({QDRANT_DIR}).")

    elif commande == "chunks":
        _afficher_chunks()

    elif commande == "compare":
        question = " ".join(args[1:]) or "combien de temps dure la surprime jeune conducteur"
        print(f"Question : {question}")
        for mode in ("dense", "sparse", "hybride"):
            print(f"\n— {mode.upper()}")
            _afficher(rechercher(question, mode=mode))

    elif commande == "eval":
        indexer()
        manquantes = verifier_questions_eval()
        if manquantes:
            print("⚠ Section attendue introuvable dans le corpus pour :")
            for q in manquantes:
                print(f"    - {q}")
            print()
        print(f"Évaluation sur {len(QUESTIONS_EVAL)} questions (hit@3 et MRR) :\n")
        for mode, s in evaluer().items():
            print(f"  {mode:<8} hit@3 = {s['hit@k']:.0%}   MRR = {s['MRR']:.2f}")
        print("\nÉchecs de l'hybride (attendu → obtenu) :")
        for q, f, mot in QUESTIONS_EVAL:
            res = rechercher(q, mode="hybride")
            if not any(r.source == f and mot in r.section.lower() for r in res):
                print(f"  ✗ {q}\n      attendu : {f} › …{mot}…")
                for r in res:
                    print(f"      obtenu  : {r.citation}")
                if not res:
                    print("      obtenu  : rien (rejeté comme hors sujet)")

        print(f"\nFilet hors sujet (rejet si dense < {SEUIL_PERTINENCE} ET BM25 < {SEUIL_SPARSE}) :")
        print("  dense   BM25   décision   question")
        for q in [x[0] for x in QUESTIONS_EVAL] + QUESTIONS_HORS_SUJET:
            d, s = scores_max(embed_question_dense(q), embed_question_sparse(q))
            rejet = d < SEUIL_PERTINENCE and s < SEUIL_SPARSE
            print(f"  {d:.2f}  {s:6.2f}   {'REJET   ' if rejet else 'accepté '}   {'(HS) ' if q in QUESTIONS_HORS_SUJET else ''}{q}")

    else:
        question = " ".join(args) or "c'est quoi le bonus malus"
        print(f"Question : {question}\n")
        _afficher(rechercher(question))