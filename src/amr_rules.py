"""Connaissances microbiologiques partagées par le pipeline et le dashboard (sans dépendance lourde)."""

# Règle catalogue : antibiotique -> (jetons de classe/sous-classe suffisants, tous requis ?)
BL_ANY = {"BETA-LACTAM", "CEPHALOSPORIN", "CARBAPENEM"}
CEPH = {"CEPHALOSPORIN", "CARBAPENEM"}
RULES = {
    "ampicillin": BL_ANY, "cefazolin": BL_ANY,
    "amoxicillin-clavulanic acid": CEPH | {"AMOXICILLIN-CLAVULANIC ACID"},
    "ampicillin-sulbactam": CEPH | {"AMOXICILLIN-CLAVULANIC ACID"},
    "piperacillin-tazobactam": CEPH | {"PIPERACILLIN-TAZOBACTAM"},
    "ticarcillin-clavulanic acid": CEPH | {"TICARCILLIN-CLAVULANIC ACID"},
    **{d: CEPH for d in ["ceftriaxone", "cefotaxime", "ceftazidime", "cefepime", "ceftiofur",
                         "cefoxitin", "aztreonam"]},
    **{d: {"CARBAPENEM"} for d in ["meropenem", "imipenem", "ertapenem", "doripenem"]},
    **{d: {"QUINOLONE"} for d in ["ciprofloxacin", "levofloxacin", "nalidixic acid"]},
    "gentamicin": {"GENTAMICIN"}, "amikacin": {"AMIKACIN"}, "tobramycin": {"TOBRAMYCIN"},
    "kanamycin": {"KANAMYCIN"}, "streptomycin": {"STREPTOMYCIN"},
    "tetracycline": {"TETRACYCLINE"}, "tigecycline": {"TIGECYCLINE"},
    "trimethoprim": {"TRIMETHOPRIM"},
    "sulfisoxazole": {"SULFONAMIDE"}, "sulfamethoxazole": {"SULFONAMIDE"},
    "trimethoprim-sulfamethoxazole": ("ALL", {"TRIMETHOPRIM"}, {"SULFONAMIDE"}),
    "chloramphenicol": {"CHLORAMPHENICOL"}, "azithromycin": {"AZITHROMYCIN"},
    "colistin": {"COLISTIN"}, "polymyxin b": {"COLISTIN"}, "nitrofurantoin": {"NITROFURAN", "NITROFURANTOIN"},
}


# Gènes intrinsèques (chromosomiques, présents chez ~100 % des souches de l'espèce) :
# exclus de la règle catalogue, sinon elle prédit "R" pour toutes les souches.
INTRINSIC_FAMILIES = {
    "A. baumannii": {"blaADC", "cxpE", "ant(3'')-IIa", "abaF", "OXA-51 family"},
    "E. coli": {"blaEC"},
    "K. pneumoniae": {"fosA", "oqxA", "oqxB"},
    "P. aeruginosa": {"blaPDC", "aph(3')-IIb", "catB7", "fosA", "OXA-50 family"},
    "S. enterica": {"aac(6')-Iaa", "aac(6')-Iy"},
}

# Jetons de classe/sous-classe utilisés comme variables "connaissances"
KNOW_TOKENS = sorted({t for r in RULES.values()
                      for t in (set().union(*r[1:]) if isinstance(r, tuple) else r)})


def rule_hit(drug: str, toks: set) -> int:
    """1 si l'ensemble de jetons (classes des gènes acquis) couvre l'antibiotique."""
    rule = RULES[drug]
    if isinstance(rule, tuple):
        return int(all(toks & grp for grp in rule[1:]))
    return int(bool(toks & rule))


def knowledge_features(gene_tokens: list[set], drug: str) -> dict:
    """Variables connaissances d'une souche : nb de déterminants acquis par classe + verdict de la règle."""
    feats = {f"k_{t}": sum(t in g for g in gene_tokens) for t in KNOW_TOKENS}
    if drug in RULES:
        feats["k_rule"] = rule_hit(drug, set().union(*gene_tokens) if gene_tokens else set())
    return feats
