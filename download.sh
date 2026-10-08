#!/usr/bin/env bash
# Télécharge les métadonnées AMR de NCBI Pathogen Detection (~1,7 Go) dans data/raw/
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p data/raw && cd data/raw
B=https://ftp.ncbi.nlm.nih.gov/pathogen/Results
get() {  # $1 = dossier NCBI, $2 = nom local
  f=$(curl -s "$B/$1/latest_snps/AMR/" | grep -o 'PDG[0-9.]*\.amr\.metadata\.tsv' | head -1)
  echo "$1 -> $f"; curl -s -o "$2" "$B/$1/latest_snps/AMR/$f"
}
get Klebsiella kleb_amr.tsv
get Escherichia_coli_Shigella ecoli_amr.tsv
get Salmonella salmonella_amr.tsv
get Acinetobacter acinetobacter_amr.tsv
get Pseudomonas_aeruginosa pseudomonas_amr.tsv
DB=https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest
curl -s -o ReferenceGeneCatalog.txt $DB/ReferenceGeneCatalog.txt
curl -s -o AMRProt-mutation.tsv $DB/AMRProt-mutation.tsv
