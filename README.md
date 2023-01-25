# Endorse - software for stochastic characterization of excavation damage zone

The software provides Byesian inversion for the excavation damage zone (EDZ) properties and stochastic contaminant transport 
in order to provide stochastic prediction of EDZ safety indicators. 

The safety indicator is defined as the 95% quantile of the contaminant concentration on the repository model boundary over the whole simulation period. 
The contaminant is modeled without radioactive decay as a inert tracer. The multilevel Monte Carlo method is 
used to parform stochastic transport simulation in reasonable time. Random inputs to the transport model 
include: EDZ parameters, random hidden fractures, (random leakage times for containers), perturbations of the rock properties.

The EDZ properties are obtained from the Bayesian inversion, using data from pore pressure min-by experiment.
The Bayesian inversion provides posterior joined probability density for the EDZ properties (porosity, permability) as heterogenous fields.
That means the properties are described as correlated random variables. 





Repository structure:

- `doc` - software documentation and various reports from the Endorse project
- `experiments` - various numerical experiments and developments as part of the Endorse project
- `src` - main sources
- `tests` - various software tests, test data





## Repository structure

- `doc` - software documentation and various reports from the Endorse project
- `experiments` - various numerical experiments and developments as part of the Endorse project
- `src` - main sources
- `tests` - various software tests, test data

## Development environment
In order to create the development environment run:

        setup.sh
        
As the Docker remote interpreter is supported only in PyCharm Proffesional, we have to debug most of the code just with
virtual environment and flow123d running in docker.
        
More complex tests should be run in the Docker image: [flow123d/geomop-gnu:2.0.0](https://hub.docker.com/repository/docker/flow123d/geomop-gnu)
In the PyCharm (need Professional edition) use the Docker plugin, and configure the Python interpreter by add interpreter / On Docker ...



## Cíl projektu

Vytvořit SW nástroj a metodiku, pro predikci veličin charakterizujících bezpečnost dílčí části úložiště
(tzv. *indikátorů bezpečnosti*) na základě geofyzikálních měření. To zahrnuje:

1. Sestavení modelu transportu kontaminace skrze EDZ od (náhodných) úložných kontejnerů do hypotetické poruchy. 
Zahrnutí předpokládané geometrie úložiště s velikostí do 100m.
2. Definice vhodných indikátorů bezpečnosti jakožto veličin odvozených od výzledků modelu transportu.
3. Tvorbu menších modelů pro identifikaci parametrů transportního modelu na základě předpokládaných průzkumů 
a geofyzikálních měření.
4. Aplikaci vhodných stochastických výpočetních metod pro predikci rozdělení indikátorů bezpečnosti a parametrů 
transportního modelu se zahrnutím chyb měření a dalších podstatných neurčitostí použitých modelů

## Rozcestník

- [Přehled řešení projektu](https://github.com/jbrezmorf/Endorse/projects/2) - přehled plánovaných, řešených a ukončených úkolů dle harmonogramu projektu

- [Přehled řešitelů](https://docs.google.com/document/d/1R8CBU9197brrruWGahVbE7_At2S2V51J6JV5bgs-kxQ/edit#heading=h.e1t1yg8nyvaz)

- [Zotero Endorse](https://www.zotero.org/groups/287302/flow123d/items/collectionKey/3BAS5Z2A) - sdílený prostor pro komunikaci referencí a fulltextů, použití v rámci aplikace [Zotero](https://www.zotero.org/download/)

- [Overleaf Endorse](https://www.overleaf.com/project) - tvorba sdílených textů, zpráv, ... 

## Software

- [Flow123d](https://github.com/flow123d/flow123d) 
 simulátor transportních a mechanických procesů v rozpukaném porézním prostředí

- [MLMC](https://github.com/GeoMop/MLMC)
  metoda multilevel Monte Carlo v Pythonu, generování náhodných polí a puklinových sítí, 
  maximal entropy method pro rekonstrukci hustoty pravděpodobnosti
  
- [PERMON](https://github.com/permon)
