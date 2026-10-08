"""Ingestion : fichiers AMR de NCBI Pathogen Detection -> tables Parquet propres.

Sources (téléchargées dans data/raw/, voir README) :
  - <pathogène>_amr.tsv           : 1 ligne = 1 génome séquencé, avec gènes AMRFinderPlus
                                    et, quand il existe, l'antibiogramme labo (AST)
  - ReferenceGeneCatalog.txt      : catalogue AMRFinderPlus gène -> classe d'antibiotique
  - AMRProt-mutation.tsv          : idem pour les mutations ponctuelles

Sorties (data/processed/) :
  - isolates.parquet   : 1 ligne par génome (espèce, pays, année, hôte, source...)
  - genes.parquet      : 1 ligne par (génome, déterminant de résistance)
  - ast.parquet        : 1 ligne par (génome, antibiotique, phénotype S/I/R)
  - gene_catalog.parquet : déterminant -> famille, classe, sous-classe
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

FILES = {
    "Klebsiella": "kleb_amr.tsv",
    "Escherichia": "ecoli_amr.tsv",
    "Salmonella": "salmonella_amr.tsv",
    "Acinetobacter": "acinetobacter_amr.tsv",
    "Pseudomonas": "pseudomonas_amr.tsv",
}

COLS = [
    "target_acc", "scientific_name", "collection_date", "geo_loc_name", "host",
    "isolation_source", "source_type", "IFSAC_category", "epi_type", "bioproject_acc",
    "number_amr_genes",
    "AST_phenotypes", "AMR_genotypes",
]

# Espèces retenues (les fichiers NCBI contiennent aussi des espèces voisines)
SPECIES = {
    "Klebsiella pneumoniae": "K. pneumoniae",
    "Escherichia coli": "E. coli",
    "Salmonella enterica": "S. enterica",
    "Acinetobacter baumannii": "A. baumannii",
    "Pseudomonas aeruginosa": "P. aeruginosa",
}

# Mutations ponctuelles sans entrée protéique dans le catalogue (mutations ADN, porines...)
POINT_PREFIX_CLASS = {
    "gyrA": ("QUINOLONE", "QUINOLONE"), "gyrB": ("QUINOLONE", "QUINOLONE"),
    "parC": ("QUINOLONE", "QUINOLONE"), "parE": ("QUINOLONE", "QUINOLONE"),
    "ompK35": ("BETA-LACTAM", "CARBAPENEM"), "ompK36": ("BETA-LACTAM", "CARBAPENEM"),
    "ompK37": ("BETA-LACTAM", "CARBAPENEM"), "oprD": ("BETA-LACTAM", "CARBAPENEM"),
    "mgrB": ("COLISTIN", "COLISTIN"), "pmrA": ("COLISTIN", "COLISTIN"),
    "pmrB": ("COLISTIN", "COLISTIN"), "phoP": ("COLISTIN", "COLISTIN"),
    "phoQ": ("COLISTIN", "COLISTIN"), "crrB": ("COLISTIN", "COLISTIN"),
    "16S": ("AMINOGLYCOSIDE", "AMINOGLYCOSIDE"), "23S": ("MACROLIDE", "MACROLIDE"),
    "ampC": ("BETA-LACTAM", "CEPHALOSPORIN"), "blaEC": ("BETA-LACTAM", "CEPHALOSPORIN"),
    "rpsL": ("AMINOGLYCOSIDE", "STREPTOMYCIN"), "folP": ("SULFONAMIDE", "SULFONAMIDE"),
    "acrB": ("EFFLUX", "EFFLUX"), "ramR": ("EFFLUX", "EFFLUX"), "marR": ("EFFLUX", "EFFLUX"),
    "soxR": ("EFFLUX", "EFFLUX"), "nfsA": ("NITROFURAN", "NITROFURANTOIN"),
    "uhpT": ("FOSFOMYCIN", "FOSFOMYCIN"), "glpT": ("FOSFOMYCIN", "FOSFOMYCIN"),
    "ptsI": ("FOSFOMYCIN", "FOSFOMYCIN"), "cyaA": ("FOSFOMYCIN", "FOSFOMYCIN"),
    "cirA": ("BETA-LACTAM", "CEFIDEROCOL"), "nfsB": ("NITROFURAN", "NITROFURANTOIN"),
    "ompF": ("BETA-LACTAM", "BETA-LACTAM"), "ompC": ("BETA-LACTAM", "BETA-LACTAM"),
    "acrR": ("EFFLUX", "EFFLUX"), "mexZ": ("EFFLUX", "EFFLUX"), "mexR": ("EFFLUX", "EFFLUX"),
    "nalC": ("EFFLUX", "EFFLUX"), "nalD": ("EFFLUX", "EFFLUX"), "adeS": ("EFFLUX", "EFFLUX"),
    "adeR": ("EFFLUX", "EFFLUX"), "dfrA51": ("TRIMETHOPRIM", "TRIMETHOPRIM"),
}

# Familles suivies pour la surveillance (préfixe de l'allèle -> libellé)
FAMILY_RULES = [
    (r"^blaKPC", "KPC"), (r"^blaNDM", "NDM"), (r"^blaOXA-48$|^blaOXA-(?:181|232|244|162|204|245|370|484|515)$", "OXA-48-like"),
    (r"^blaVIM", "VIM"), (r"^blaIMP", "IMP"), (r"^blaOXA-(?:23|24|40|58|72|143)", "OXA-23/24/58 (Acineto.)"),
    (r"^blaCTX-M", "CTX-M (BLSE)"), (r"^mcr-", "mcr (colistine)"),
    (r"^armA|^rmt", "16S méthylase (aminosides)"), (r"^tet\(X", "tet(X) (tigécycline)"),
]


def country_of(geo: pd.Series) -> pd.Series:
    c = geo.fillna("").str.split(":").str[0].str.strip()
    c = c.replace({"": None, "missing": None, "not collected": None, "Not collected": None,
                   "not provided": None, "Not Provided": None, "Not applicable": None,
                   "unknown": None, "Unknown": None, "Viet Nam": "Vietnam"})
    return c


def year_of(date: pd.Series) -> pd.Series:
    y = pd.to_numeric(date.fillna("").str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")
    return y.where((y >= 1950) & (y <= 2026)).astype("Int16")


def reservoir_of(df: pd.DataFrame) -> pd.Series:
    """Réservoir d'origine : hôte déclaré, sinon source d'isolement / catégorie IFSAC."""
    host = df["host"].fillna("").str.lower()
    src = (df["isolation_source"].fillna("") + " | " + df["IFSAC_category"].fillna("")).str.lower()
    stype = df["source_type"].fillna("").str.lower()
    out = pd.Series("Non renseigné", index=df.index)
    rules = [  # du moins prioritaire au plus prioritaire
        (src, r"water|soil|sewage|wastewater|environment|surface|river|swab of|sponge|drag", "Environnement"),
        (src, r"food|feed|nut|spice|produce|vegetable|fruit|egg|fish|seafood|shrimp|cheese|milk|meal", "Aliment / autre"),
        (src, r"canine|dog|feline|cat|equine|horse|pet", "Animal de compagnie"),
        (src, r"pork|swine|porcine|pig", "Porc"),
        (src, r"beef|cattle|bovine|veal|dairy cow", "Bovin"),
        (src, r"chicken|poultry|turkey|broiler|duck|hatchery", "Volaille"),
        (host, r"canis|dog|felis|cat|equus|horse", "Animal de compagnie"),
        (host, r"sus scrofa|swine|pig|porcine", "Porc"),
        (host, r"bos taurus|cattle|cow|bovine|calf", "Bovin"),
        (host, r"gallus|chicken|poultry|broiler|turkey|meleagris|duck|anas ", "Volaille"),
        (stype, r"^human$", "Humain"),
        (host, r"homo sapiens|human|patient", "Humain"),
    ]
    for col, pat, lab in rules:
        out[col.str.contains(pat, regex=True)] = lab
    return out


def load_catalog() -> pd.DataFrame:
    cat = pd.read_csv(RAW / "ReferenceGeneCatalog.txt", sep="\t", dtype=str)
    cat["symbol"] = cat["allele"].fillna(cat["gene_family"])
    a = cat[["symbol", "gene_family", "class", "subclass", "type"]]
    a = a[a["type"].isin(["AMR"])].drop(columns="type")
    mut = pd.read_csv(RAW / "AMRProt-mutation.tsv", sep="\t", dtype=str)
    b = mut.rename(columns={"standard_mutation_symbol": "symbol"})[["symbol", "class", "subclass"]]
    b["gene_family"] = b["symbol"].str.split("_").str[0]
    fam = cat[["gene_family", "class", "subclass"]].dropna().drop_duplicates("gene_family")
    fam = fam.rename(columns={"gene_family": "symbol"})
    fam["gene_family"] = fam["symbol"]
    return pd.concat([a, b, fam]).drop_duplicates("symbol")


def annotate(genes: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    g = genes.merge(catalog, how="left", left_on="gene", right_on="symbol").drop(columns="symbol")
    miss = g["class"].isna()
    prefix = g.loc[miss, "gene"].str.split("_").str[0]
    g.loc[miss, "class"] = prefix.map(lambda p: POINT_PREFIX_CLASS.get(p, (None, None))[0])
    g.loc[miss, "subclass"] = prefix.map(lambda p: POINT_PREFIX_CLASS.get(p, (None, None))[1])
    g["gene_family"] = g["gene_family"].fillna(g["gene"].str.split("_").str[0])
    g["class"] = g["class"].fillna("AUTRE")
    g["subclass"] = g["subclass"].fillna("AUTRE")
    g["watch_family"] = None
    for pat, lab in FAMILY_RULES:
        m = g["watch_family"].isna() & g["gene"].str.contains(pat, regex=True)
        g.loc[m, "watch_family"] = lab
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    iso_parts, gene_parts, ast_parts = [], [], []
    for genus, fname in FILES.items():
        print(f"lecture {fname} ...")
        df = pd.read_csv(RAW / fname, sep="\t", usecols=COLS, dtype=str)
        df["species"] = df["scientific_name"].str.extract(r"^(\S+ \S+)")[0].map(SPECIES)
        df = df[df["species"].notna()].copy()
        iso = pd.DataFrame({
            "isolate": df["target_acc"],
            "species": df["species"],
            "year": year_of(df["collection_date"]),
            "country": country_of(df["geo_loc_name"]),
            "reservoir": reservoir_of(df),
            "epi_type": df["epi_type"].fillna("non renseigné"),
            "bioproject": df["bioproject_acc"],
            "n_amr_genes": pd.to_numeric(df["number_amr_genes"], errors="coerce").astype("Int16"),
            "has_ast": df["AST_phenotypes"].notna(),
        })
        iso_parts.append(iso)

        # Gènes : "blaKPC-2,gyrA_S83I=POINT,blaOXA=MISTRANSLATION,..."
        g = df[["target_acc", "AMR_genotypes"]].dropna()
        g = g.assign(token=g["AMR_genotypes"].str.split(",")).explode("token")
        g["gene"] = g["token"].str.split("=").str[0].str.strip()
        g["tag"] = g["token"].str.split("=").str[1].fillna("COMPLETE")
        # on écarte les hits non fonctionnels / incertains
        g = g[~g["tag"].isin(["MISTRANSLATION", "HMM", "PARTIAL_END_OF_CONTIG", "INTERNAL_STOP"])]
        gene_parts.append(pd.DataFrame({
            "isolate": g["target_acc"].values, "gene": g["gene"].values,
            "is_point": (g["tag"] == "POINT").values,
        }).drop_duplicates(["isolate", "gene"]))

        # AST : "amikacin=S,aztreonam=R,..."
        a = df[["target_acc", "AST_phenotypes"]].dropna()
        a = a.assign(tok=a["AST_phenotypes"].str.split(",")).explode("tok")
        kv = a["tok"].str.rsplit("=", n=1, expand=True)
        ast_parts.append(pd.DataFrame({
            "isolate": a["target_acc"].values,
            "antibiotic": kv[0].str.strip().str.lower().values,
            "phenotype": kv[1].str.strip().values,
        }))
        print(f"  {genus}: {len(iso):,} génomes, {iso['has_ast'].sum():,} avec antibiogramme")

    isolates = pd.concat(iso_parts, ignore_index=True).drop_duplicates("isolate")
    catalog = load_catalog()
    genes = annotate(pd.concat(gene_parts, ignore_index=True), catalog)
    ast = pd.concat(ast_parts, ignore_index=True)
    # phénotypes normalisés (S / I / R ; SDD = sensible dose-dépendant -> I ; NS -> R)
    ast["phenotype"] = ast["phenotype"].replace({"SDD": "I", "NS": "R", "ND": None, "not defined": None})
    ast = ast[ast["phenotype"].isin(["S", "I", "R"])].drop_duplicates(["isolate", "antibiotic"])

    isolates.to_parquet(OUT / "isolates.parquet", index=False)
    genes.to_parquet(OUT / "genes.parquet", index=False)
    ast.to_parquet(OUT / "ast.parquet", index=False)
    catalog.to_parquet(OUT / "gene_catalog.parquet", index=False)
    print(f"\n{len(isolates):,} génomes | {len(genes):,} déterminants | {len(ast):,} résultats d'antibiogramme")


if __name__ == "__main__":
    main()
