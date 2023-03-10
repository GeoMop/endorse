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



## C??l projektu

Vytvo??it SW n??stroj a metodiku, pro predikci veli??in charakterizuj??c??ch bezpe??nost d??l???? ????sti ??lo??i??t??
(tzv. *indik??tor?? bezpe??nosti*) na z??klad?? geofyzik??ln??ch m????en??. To zahrnuje:

1. Sestaven?? modelu transportu kontaminace skrze EDZ od (n??hodn??ch) ??lo??n??ch kontejner?? do hypotetick?? poruchy. 
Zahrnut?? p??edpokl??dan?? geometrie ??lo??i??t?? s velikost?? do 100m.
2. Definice vhodn??ch indik??tor?? bezpe??nosti jako??to veli??in odvozen??ch od v??zledk?? modelu transportu.
3. Tvorbu men????ch model?? pro identifikaci parametr?? transportn??ho modelu na z??klad?? p??edpokl??dan??ch pr??zkum?? 
a geofyzik??ln??ch m????en??.
4. Aplikaci vhodn??ch stochastick??ch v??po??etn??ch metod pro predikci rozd??len?? indik??tor?? bezpe??nosti a parametr?? 
transportn??ho modelu se zahrnut??m chyb m????en?? a dal????ch podstatn??ch neur??itost?? pou??it??ch model??

## Rozcestn??k

- [P??ehled ??e??en?? projektu](https://github.com/jbrezmorf/Endorse/projects/2) - p??ehled pl??novan??ch, ??e??en??ch a ukon??en??ch ??kol?? dle harmonogramu projektu

- [P??ehled ??e??itel??](https://docs.google.com/document/d/1R8CBU9197brrruWGahVbE7_At2S2V51J6JV5bgs-kxQ/edit#heading=h.e1t1yg8nyvaz)

- [Zotero Endorse](https://www.zotero.org/groups/287302/flow123d/items/collectionKey/3BAS5Z2A) - sd??len?? prostor pro komunikaci referenc?? a fulltext??, pou??it?? v r??mci aplikace [Zotero](https://www.zotero.org/download/)

- [Overleaf Endorse](https://www.overleaf.com/project) - tvorba sd??len??ch text??, zpr??v, ... 

## Software

- [Flow123d](https://github.com/flow123d/flow123d) 
 simul??tor transportn??ch a mechanick??ch proces?? v rozpukan??m por??zn??m prost??ed??

- [MLMC](https://github.com/GeoMop/MLMC)
  metoda multilevel Monte Carlo v Pythonu, generov??n?? n??hodn??ch pol?? a puklinov??ch s??t??, 
  maximal entropy method pro rekonstrukci hustoty pravd??podobnosti
  
- [PERMON](https://github.com/permon)
