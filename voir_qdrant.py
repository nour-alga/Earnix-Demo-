from src.rag import client_qdrant, COLLECTION

client = client_qdrant()

# 1. Infos sur la collection
info = client.get_collection(COLLECTION)
print("Nombre de points :", info.points_count)
print("Vecteurs denses  :", info.config.params.vectors)
print("Vecteurs sparse  :", info.config.params.sparse_vectors)

# 2. Parcourir les 5 premiers points
points, _ = client.scroll(COLLECTION, limit=5, with_payload=True, with_vectors=True)
for p in points:
    print("\n" + "-" * 60)
    print("ID       :", p.id)
    print("Citation :", p.payload["citation"])
    print("Texte    :", p.payload["texte"][:100])
    print("Dense    :", len(p.vector["dense"]), "valeurs, début :", [round(x, 3) for x in p.vector["dense"][:5]])
    sp = p.vector["sparse"]
    print("Sparse   :", len(sp.indices), "mots non nuls, ex :", list(zip(sp.indices[:5], [round(v, 2) for v in sp.values[:5]])))