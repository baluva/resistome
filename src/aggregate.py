"""Agrégats de surveillance (petits fichiers versionnés, lus par le dashboard).

Attention à l'interprétation : NCBI Pathogen Detection agrège les génomes *déposés*,
pas un échantillonnage représentatif. Les "prévalences" sont des parts parmi les
génomes séquencés (biais vers les souches résistantes, les épidémies, les pays riches
en séquençage). On les lit comme des tendances relatives, pas comme des taux cliniques.
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PROC = (ROOT / "data" / "processed").as_posix()
APP = ROOT / "data" / "app"

ISO3 = """
select * from (values
 ('USA','USA'),('United Kingdom','GBR'),('China','CHN'),('Germany','DEU'),('France','FRA'),
 ('Canada','CAN'),('Australia','AUS'),('Italy','ITA'),('Spain','ESP'),('Netherlands','NLD'),
 ('India','IND'),('Brazil','BRA'),('Japan','JPN'),('South Korea','KOR'),('Thailand','THA'),
 ('Vietnam','VNM'),('Pakistan','PAK'),('Bangladesh','BGD'),('Nigeria','NGA'),('Kenya','KEN'),
 ('South Africa','ZAF'),('Egypt','EGY'),('Tunisia','TUN'),('Morocco','MAR'),('Algeria','DZA'),
 ('Mexico','MEX'),('Argentina','ARG'),('Chile','CHL'),('Colombia','COL'),('Peru','PER'),
 ('Russia','RUS'),('Turkey','TUR'),('Iran','IRN'),('Saudi Arabia','SAU'),('Israel','ISR'),
 ('Greece','GRC'),('Portugal','PRT'),('Belgium','BEL'),('Switzerland','CHE'),('Austria','AUT'),
 ('Denmark','DNK'),('Sweden','SWE'),('Norway','NOR'),('Finland','FIN'),('Ireland','IRL'),
 ('Poland','POL'),('Czech Republic','CZE'),('Czechia','CZE'),('Hungary','HUN'),('Romania','ROU'),
 ('Serbia','SRB'),('Croatia','HRV'),('Ukraine','UKR'),('Lebanon','LBN'),('Jordan','JOR'),
 ('Iraq','IRQ'),('Kuwait','KWT'),('Qatar','QAT'),('United Arab Emirates','ARE'),('Oman','OMN'),
 ('Singapore','SGP'),('Malaysia','MYS'),('Indonesia','IDN'),('Philippines','PHL'),('Taiwan','TWN'),
 ('Hong Kong','HKG'),('Nepal','NPL'),('Sri Lanka','LKA'),('Cambodia','KHM'),('Laos','LAO'),
 ('Myanmar','MMR'),('Ghana','GHA'),('Ethiopia','ETH'),('Tanzania','TZA'),('Uganda','UGA'),
 ('Malawi','MWI'),('Mozambique','MOZ'),('Cameroon','CMR'),('Senegal','SEN'),('Burkina Faso','BFA'),
 ('Mali','MLI'),('Niger','NER'),('Sudan','SDN'),('Democratic Republic of the Congo','COD'),
 ('Rwanda','RWA'),('Zambia','ZMB'),('Zimbabwe','ZWE'),('Madagascar','MDG'),('Gambia','GMB'),
 ('Ecuador','ECU'),('Venezuela','VEN'),('Uruguay','URY'),('Paraguay','PRY'),('Bolivia','BOL'),
 ('Guatemala','GTM'),('Cuba','CUB'),('Dominican Republic','DOM'),('Haiti','HTI'),('Puerto Rico','PRI'),
 ('New Zealand','NZL'),('Estonia','EST'),('Latvia','LVA'),('Lithuania','LTU'),('Slovenia','SVN'),
 ('Slovakia','SVK'),('Bulgaria','BGR'),('Albania','ALB'),('Kazakhstan','KAZ'),('Georgia','GEO'),
 ('Armenia','ARM'),('Libya','LBY'),('Yemen','YEM'),('Syria','SYR'),('Afghanistan','AFG'),
 ('Mongolia','MNG'),('Côte d''Ivoire','CIV'),('Cote d''Ivoire','CIV'),('Benin','BEN'),('Togo','TGO'),
 ('Guinea','GIN'),('Sierra Leone','SLE'),('Liberia','LBR'),('Gabon','GAB'),('Botswana','BWA'),
 ('Namibia','NAM'),('Iceland','ISL'),('Luxembourg','LUX'),('Cyprus','CYP'),('Malta','MLT')
) t(country, iso3)
"""


def main() -> None:
    APP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql(f"""
        create view i as select * from '{PROC}/isolates.parquet';
        create view g as select * from '{PROC}/genes.parquet';
        create view a as select * from '{PROC}/ast.parquet';
        create table iso3 as {ISO3};
        -- 1 ligne par (génome, famille surveillée)
        create table fam as select distinct isolate, watch_family from g where watch_family is not null;
    """)

    def save(name: str, sql: str) -> None:
        df = con.sql(sql).df()
        df.to_parquet(APP / f"{name}.parquet", index=False)
        print(f"{name:22s} {len(df):6d} lignes")

    save("kpis", """
        select species, count(*) n_genomes, sum(has_ast::int) n_ast,
               count(distinct country) n_countries, min(year) y_min, max(year) y_max,
               avg(n_amr_genes) mean_amr_genes
        from i group by 1 order by 2 desc""")

    # Part des génomes porteurs de chaque famille, par espèce x année
    save("family_year", """
        with base as (select species, year, count(*) n from i where year between 2005 and 2025 group by all)
        select i.species, i.year, f.watch_family, count(*) n_pos, any_value(b.n) n,
               count(*) / any_value(b.n) as share
        from fam f join i using(isolate) join base b on b.species=i.species and b.year=i.year
        where i.year between 2005 and 2025
        group by all""")
    save("year_totals", "select species, year, count(*) n from i where year between 2005 and 2025 group by all")

    # Carte : part des génomes porteurs, par pays (pays avec >= 50 génomes de l'espèce)
    save("family_country", """
        with base as (select species, country, count(*) n from i where country is not null group by all having n >= 50)
        select b.species, b.country, iso3.iso3, f.watch_family, count(f.isolate) n_pos, any_value(b.n) n,
               count(f.isolate) / any_value(b.n) as share
        from base b join i on i.species=b.species and i.country=b.country
        left join fam f using(isolate)
        left join iso3 on iso3.country = b.country
        where f.watch_family is not null
        group by all""")
    save("country_totals", """
        select species, i.country, iso3.iso3, count(*) n, avg(n_amr_genes) mean_amr_genes
        from i left join iso3 using(country) where i.country is not null group by all""")

    # Réservoirs : One Health (humain / animal / aliment / environnement)
    save("family_reservoir", """
        with base as (select species, reservoir, count(*) n from i group by all)
        select b.species, b.reservoir, f.watch_family, count(f.isolate) n_pos, any_value(b.n) n,
               count(f.isolate) / any_value(b.n) as share
        from base b join i on i.species=b.species and i.reservoir=b.reservoir
        join fam f using(isolate) group by all""")
    save("reservoir_totals", "select species, reservoir, count(*) n, avg(n_amr_genes) mean_amr_genes from i group by all")

    # Classes de résistance portées (part des génomes avec >= 1 déterminant de la classe)
    save("class_share", """
        with base as (select species, count(*) n from i group by 1),
        c as (select distinct isolate, class from g where class not in ('AUTRE','EFFLUX'))
        select i.species, c.class, count(*) n_pos, any_value(b.n) n, count(*)/any_value(b.n) as share
        from c join i using(isolate) join base b on b.species=i.species
        group by all having share >= 0.01""")

    # Antibiogrammes labo : % R par espèce x antibiotique
    save("ast_summary", """
        select i.species, a.antibiotic, count(*) n,
               avg((phenotype='R')::int) pct_r, avg((phenotype='I')::int) pct_i
        from a join i using(isolate) group by all having n >= 100""")

    # Multirésistance : nb de classes portées par génome (distribution)
    save("mdr", """
        with c as (select isolate, count(distinct class) n_classes from g
                   where class not in ('AUTRE','EFFLUX') and not is_point group by 1)
        select i.species, least(coalesce(c.n_classes,0), 10) n_classes, count(*) n
        from i left join c using(isolate) group by all""")

    # Gènes les plus fréquents par espèce (acquis, hors mutations ponctuelles et efflux)
    save("top_genes", """
        with base as (select species, count(*) n from i group by 1)
        select i.species, g.gene, g.class, g.subclass, count(*) n_pos, count(*)/any_value(b.n) as share
        from g join i using(isolate) join base b on b.species=i.species
        where not g.is_point and g.class not in ('AUTRE','EFFLUX')
        group by i.species, g.gene, g.class, g.subclass qualify row_number() over (partition by i.species order by count(*) desc) <= 25""")

    # Petit échantillon de profils réels pour la page "Prédire" (souches avec antibiogramme)
    save("demo_isolates", """
        select i.isolate, i.species, i.country, i.year, i.reservoir,
               string_agg(g.gene, ',' order by g.gene) genes
        from i join g using(isolate)
        where i.has_ast group by all""")


if __name__ == "__main__":
    main()
