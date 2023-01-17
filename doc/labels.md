# Annotations

## Répertoire d’immeubles localisés (RIL)

Le RIL contient la liste des adresses d’habitation des communes de 10 000 habitants ou plus et **de la plupart des communes de moins de 10 000 habitants dans les DOM.** Seules les communes des Antilles-Guyane de Sainte-Julie, Saül, Camopi et Ouanary n’ont pas de RIL.

Informations utiles :

- Un élément du RIL correspond à la plus petite unité contenant des logements repérable sur le terrain à partir de son adresse. La plupart du temps, il s’agit d'un bâtiment associé à une adresse. Il peut également s’agir de deux bâtiments ou plus s’il n’est pas possible de les distinguer sur le terrain (par exemple, via un affichage "bâtiment A", "B" visible sur le terrain)
- Informations contenues dans le RIL pour chaque adresse :
    - un type (avenue, boulevard, rue, impasse, chemin, sente, etc.) et un nom de voie ;
    - un numéro dans la voie, éventuellement un suffixe (bis, ter, A, B, etc.) ;
    - un complément d’adresse (lieu-dit, numéro de bâtiment, etc.)
    - des coordonnées géographiques X et Y (cf. 2.3 pour la précision de ces coordonnées) ;
- Pour chaque ensemble immobilier :
    - la catégorie (habitation, communauté, établissement touristique) ;
    - le type (maison individuelle, ensemble de maisons individuelles, immeuble, etc.) ;
    - l’actualité (habitable, en construction, périmé) ;
    - le nombre de logements ;
    - la date de construction ;
    - la date d’entrée dans le répertoire ou la date de dernière modification ;
    - le numéro de permis de construire ;
    - le numéro de parcelle cadastrale.
- Référentiels géographiques des coordonnées :
    - UTM zone 20N en Guadeloupe et Martinique (code EPSG : 4559)
    - UTM zone 22N en Guyane (code EPSG : 2972)
    - UTM zone 38S à Mayotte (code EPSG : 4471)
    - UTM zone 40S à La Réunion (code EPSG : 2975)

### Processus de mise à jour en Antilles-Guyane :

Aux Antilles-Guyane, pour les habitations et les établissements touristiques, une campagne de mise à jour du RIL se déroule de novembre à octobre. Le RIL est mis à jour principalement par une enquête annuelle sur le terrain, appelée enquête cartographique, mais aussi à partir des permis de construire. Le RIL est également mis à jour avec les résultats du recensement, par les fiches navettes de novembre à mai et par les résultats de collecte à partir de mai. Le CorRIL doit ensuite expertiser le RIL avant le 30 septembre pour que les RIL soient livrés par les établissements régionaux de l’Insee dans leur version définitive avant le 31 octobre.

#### Enquête cartographique

L’enquête cartographique se déroule sur le terrain de fin avril à mi-août, avec un traitement par les gestionnaires jusqu’à fin août. Des enquêteurs de l’Insee mettent à jour les îlots concernés par la collecte de l’année suivante (en petite commune, tous les îlots appartiennent au même groupe de rotation). Par exemple, cette enquête a porté en 2021 sur les îlots du groupe de rotation n°4, dans lesquels ont été tirées les adresses collectées en 2022. Elle met également à jour les communautés. Elle est quasi-exhaustive, c’est donc la principale source de mise à jour du RIL en Antilles-Guyane. Seuls quelques îlots isolés sont enquêtés uniquement lors de la tournée de reconnaissance de la collecte (enquête dite « 2 en 1 »). De plus, certains îlots désignés par les communes comme n’ayant pas évolué depuis 5 ans ne sont pas enquêtés exhaustivement. Le RIL est donc mis à jour chaque année sur un cinquième du territoire de la commune, ce qui fait qu’il est plus ou moins à jour selon les îlots.

#### Permis de construire

Les permis sont fournis par le ministère chargé du logement et permettent aux équipes recensement de mettre à jour le RIL suite aux constructions de logements. Les permis sont livrés 4 fois par campagne (en novembre, février, avril et août). Les nouvelles constructions repérées lors du traitement des permis, ainsi que certaines habitations dont on veut vérifier l’état, sont envoyées aux mairies via dans des listes **d’entités adressées à confirmer** (EAAC). Les mairies peuvent alors répondre aux listes en signalant quelles constructions sont devenues habitables et peuvent être prises en compte pour le recensement. C’est la commune qui choisit le nombre d’envois qu’elle souhaite recevoir, de 1 à 4.

#### Intégration des résultats du recensement

#### Expertise des communes

#### Livraison du RIL

La livraison des RIL marque la fin de la campagne. La livraison des RIL fait suite à l’expertise légale réalisée par les communes avant le 30 septembre. Avant de procéder à la livraison des RIL, les équipes recensement doivent justifier les fortes évolutions du nombre de logements habitables des îlots comprenant plus de 100 logements et des communes.

Première livraison des RIL entre mi-septembre et mi-octobre. Les RIL livrés sont contrôlés par l’Insee. Ces contrôles peuvent conduire à des suspensions qui nécessiteront de nouvelles livraisons des RIL avec une échéance fixée au 31 octobre. Quand les RIL livrés sont validées, l’Insee constitue les bases de sondages d’adresses et procède au tirage de l’échantillon des adresses à recenser pour l’enquête annuelle de recensement à venir.

#### Mise à jour des communautés

Intégration des fichiers administratifs. Les équipes recensement reçoivent, d’avril à juin, des listes d’ouvertures et de fermetures potentielles de communautés après traitement par l’Insee de fichiers administratifs provenant par exemple du Ministère de l’Éducation pour les établissements scolaires ou du Ministère des Solidarités et de la Santé pour les établissements sanitaires et sociaux.

### Processus de mise à jour à la Réunion

Similaire au processus Antilles-Guyane, campagne de mise à jour du RIL de novembre à octobre. Le RIL est mis à jour principalement par les permis de construire, les données fiscales et une enquête annuelle sur le terrain pour vérifier l’état des habitations en construction et de certaines parcelles cadastrales, appelée enquête cartographique en différentiel.

De 2011 à 2016, la principale source de mise à jour était l’enquête cartographique en différentiel, qui portait uniquement sur un seul groupe de rotation. Depuis 2017, les données fiscales sont mobilisées en complément de l’enquête cartographique en différentiel. Un rattrapage est effectué pour pouvoir mettre à jour tous les groupes de rotation chaque année à partir de 2023.

#### Les enquêtes terrain

**L’enquête cartographique en différentiel** est préparée à partir de mars et se déroule sur le terrain de mai à décembre. Les enquêteurs examinent des habitations en construction et leurs alentours pour voir si elles sont devenues habitables. Ils examinent également les habitations repérées grâce aux données fiscales. Pour cela, des gestionnaires traitent également des EA ou des parcelles cadastrales sur lesquelles on soupçonne un déficit dans le RIL. Elles sont repérées à la suite d’un appariement entre le RIL et des données fiscales (Fichiers démographiques sur les logements et les individus, qui apparie notamment des sources fiscales), réalisé par l’Insee. Les données fiscales constituent donc une source complémentaire à celle des permis pour améliorer la qualité du RIL. 

**L’enquête de mesure de la qualité du RIL** (EMQR) à La Réunion se déroule sur le terrain d’octobre à décembre. Les enquêteurs effectuent un ratissage complet d’îlots du groupe de rotation. L’objectif est d’évaluer la qualité du RIL sur ce groupe de rotation et de voir des
adresses habitables qui n’ont pas été repérées lors de l’enquête cartographique en différentiel.

### Processus de mise à jour à Mayotte

Identique au processus pour Antilles-Guyane.

### Dictionnaire

| Variable  | Longueur          | Description |
| :-------- |:-----------------:| :---------- |
| NumVoie | 4          | Numéro dans la voie (0 pour les adresses non normalisées) |
| Suffixe | 4          | Indice de répétition ou suffixe : BIS, TER, QUA...ou encore A, B... |
| TypeVoie | 4          | Type de voie selon norme DGFiP |
| LibVoie | 32          | Libellé de la voie |
| Rivoli | 7          | Code Rivoli de la voie |
| Complément | 32          | Complément d’adresse |
| Statut | 3          | Catégorie et actualité : habitation habitable (HAB) ou en cours de construction (EC), communauté (CMT), établissement touristique (HOT) |
| ID_Externe | 17          | N° de Permis de construire ou identifiant de la communauté au répertoire des communautés ou identifiant du dernier recensement exhaustif (soit en général celui de 1999) ou à blanc |
| NbLogement | 4          | Nombre de logements habitables à cette adresse (pour les adresses d’habitation, et mis à zéro pour les accès secondaires) |
| MAD_REFCAD | 10          | Référence cadastrale |
| Principal | 1         | 'O' pour OUI s’il s’agit d’un accès principal 'N' pour NON s’il s’agit d’un accès secondaire |
| iris2008 | 4          | Code IRIS |
| Id_ea(*) | 15          | Identifiant Insee pour le RIL |
| X | 10          | Valeur de l’abscisse exprimée avec 2 décimales (dans la projection choisie) |
| Y | 10          | Valeur de l’ordonnée exprimée avec 2 décimales (dans la projection choisie) |
| depcom | 5          | Code commune |

## BD TOPO

Documentation générale : 

Les évolutions du produit sont décrites dans des documents spécifiques nommés :

- « Suivi des évolutions à partir de la version 3.0 » ;
- « Suivi des évolutions de la version 1.0 à la version 3.0 » ;
- « Suivi des évolutions du descriptif de livraison ».

La BD TOPO est la modélisation 3D du territoire et de ses infrastructures. Précision métrique. Les objets de la BD TOPO sont regroupés par thèmes guidés :

- Limites et unités administratives
- Adresses postales
- Constructions
- Éléments ayant trait à l’eau
- Lieu ou lieu-dit possédant un toponyme et décrivant un espace naturel ou un lieu habité
- Végétation, estran
- Services publics, stockage et transport des sources d'énergie, lieux et sites industriels
- Infrastructures du réseau routier, ferré et aérien
- Zonages faisant l'objet de réglementations spécifiques

### Mise à jour

L’actualité des informations est définie par le décalage entre les données de la BD TOPO (J+1) et le terrain nominal à une date T. L’actualité décrit la « fraîcheur » des données. Le rythme de disponibilité peut varier :

- J+1
- trimestriel ;
- annuel.

De nombreux thèmes de la BD TOPO sont mis à jour en continu ; leur exhaustivité est consolidée lors des processus cycliques variant selon les thèmes : restitution photogrammétrique, exploitation de différentiels issus de sources partenariales. La mise à jour du thème bâti suit le cycle de mise à jour des prises de vues aériennes (de 3 à 4 ans) mais des mises à jour intermédiaires pour les bâtiments sont possibles, notamment à partir du cadastre.

### Enrichissement

Le thème Bâti qui intègre progressivement depuis 2008 les bâtiments du cadastre, donnant à ce thème une granularité plus importante, c'est-à-dire une description plus fine des bâtiments (contour plus précis, découpage plus important). Cette intégration se fait au fur et à
mesure : la BD TOPO actuelle fait coexister des bâtiments initialement dans la BD TOPO et des bâtiments intégrés à partir des données du cadastre.

### Exhaustivité

Exigence de 95% sur le bâti.

### Variables

Voir le document sur le site de l'IGN. Variables qui peuvent nous intéresser :

- Date d'apparition, de confirmation, de création ;
- Etat de l'objet ;
- Fictif ;
- Graphie du toponyme ;
- Nature de l'objet ;
- Toponyme ;

Liste des classes pour le bâti :

- BATIMENT : classe qui nous intéresse
- CIMETIERE
- CONSTRUCTION LINEAIRE
- CONSTRUCTION PONCTUELLE
- CONSTRUCTION SURFACIQUE
- LIGNE OROGRAPHIQUE
- PYLONE
- RESERVOIR
- TERRAIN DE SPORT
- TOPONYMIE BATI

Informations sur les bâtiments :

Définition : Construction au-dessus du sol qui est utilisée pour abriter des humains, des animaux, des objets, pour la production de biens économiques ou pour la prestation de services et qui se réfère à toute structure construite ou érigée de façon permanente sur son site.
Sélection : Initialement, les seuils de sélection des bâtiments étaient les suivants :

- Tous les bâtiments de plus de 50 m² sont inclus.
- Les bâtiments faisant entre 20 et 50 m² sont sélectionnés en fonction de leur environnement et de leur aspect.
- Les bâtiments de moins de 20 m² sont représentés par un objet de classe Construction ponctuelle s’ils sont très hauts, ou s’ils sont spécifiquement désignés sur la carte au 1 : 25 000 en cours (ex: antenne, transformateur...).

Après unification de la BD TOPO avec la BD PARCELLAIRE, tous les bâtiments présents dans la dernière édition de la BD PARCELLAIRE® vecteur sont inclus, sauf éventuellement des bâtiments manifestement détruits depuis la date de validité de la BD PARCELLAIRE. Les petits bâtiments de la BD PARCELLAIRE qui représentent des constructions ponctuelles (exemple des transformateurs) ou des constructions linéaires (exemple des murs de remparts) sont saisis avec leur modélisation initiale respective en BD TOPO. Il n’existe plus de seuil minimal pour la superficie des bâtiments.

Cependant, si une nouvelle saisie photogrammétrique a lieu après les phases d’unification du bâti, les nouveaux bâtiments ne posséderont pas la granularité de la BD PARCELLAIRE. Pour la restitution, les seuils de sélection initiaux sont alors appliqués (bâtiments de plus de 50 m² et bâtiments de 20 à 50 m² en fonction de leur environnement et de leur aspect).

Dans les natures proposées pour la classe *bâtiment* : 

- Nature *indifférenciée* : c'est la valeur prise par défaut, chaque fois que l'aspect général d'un bâtiment ne révèle rien de sa
nature exacte. Regroupement : Bâtiment d'habitation | Bâtiments administratifs | Bâtiment public | Bergerie traditionnelle
(bâtiment) | Borie | Bungalow | Bureaux | Chalet | Grange (bâtiment) | Immeuble collectif d'habitation | Immeuble |
Lavoir couvert | Maison | Refuge (bâtiment) | Ferme | Garage individuel | Gymnase (bâtiment) | Gare téléphérique
ou télésiège (bâtiment) | Aérogare (bâtiment) | Gare (bâtiment). Tous les bâtiments en dur dont l'architecture ou l'aspect n’est
pas industriel, agricole ou commercial
- Nature *industriel, agricole ou commercial*

Peut-être que le thème **Lieu ou lieu-dit possédant un toponyme et décrivant un espace naturel ou un lieu habité** peut aussi nous intéresser, notamment pour la classe **zone d'habitation** : "Lieu-dit habité de manière permanente, temporaire mais régulière ou anciennement habité et à l'état de ruines." Notes : A de rares exceptions (habitations de Guyane, site ruiné...), une zone d’habitation doit posséder un toponyme. Le calcul des emprises des zones d'habitation est différent en Guyane où il est issu des zones d'occupation du sol produites dans le cadre du RGG (Référentiel Géographique Guyanais).

Nature particulière : **Habitat temporaire**. Définition : En Guyane française, lieu d'habitat intermittent nommé, régulièrement fréquenté par les habitants ou exploitants de la forêt. Regroupement : Carbet | Habitation intermittente (Guyane) | Maison communautaire (Guyane)
Sélection : Les carbets (maisons communautaires) habitées de manière intermittente au gré des déplacements des habitants sont retenus s'ils possèdent un toponyme. Valeurs du champ « Nature détaillée » associées : Sans valeur | Carbet.

### Livraisons

Livraisons par département (avec un buffer de 5 km) d'outre-mer disponibles. PostgreSQL (.sql), GeoPackage (.gpkg) ou Shapefile.

Volume approximatif : 2 Go par département pour la BD TOPO complète. Fichier BATIMENT qui nous intéresse a priori ainsi que le fichier ZONE_D_HABITATION.

Métadonnées et suppléments dans des fichiers annexes.

Fichiers sur le portail [géoservices](https://geoservices.ign.fr/documentation/donnees/vecteur/bdtopo)

### Versions

Numéro de version : X.Y, par exemple 3.0 -> 3 est le numéro de version, 0 de sous-version. 

Les classes d’objets peuvent, dans de rares cas; être amenées à changer pour être ajoutées, supprimées, ou modifiées. Dans le cas d’un ajout ou d’une suppression de classe, il s’agit d’une modification de structure d’un thème. Ces changements incrémentent le numéro de sous-version du produit.

Les thèmes peuvent, exceptionnellement; être amenés à changer pour être ajoutés, supprimés, ou modifiés. Dans le cas d’un ajout ou d’une suppression de thème, il s’agit d’une modification de structure du produit. Ces changements incrémentent le numéro de sous-version du produit.

Le numéro de version du produit reste le même si son processus de fabrication est inchangé. À partir de l’édition de mars 2023 (23.1), l’arborescence des données livrées ne contient, quant à elle, que le numéro de version, pas le numéro de sous-version.

La BD TOPO est éditée quatre fois par an.

Pas de gros changements en version 3, qui a débuté en mars 2019. Version 3.0 Bêta à partir d'octobre 2018. Différences avec la version 2.2 antérieure : https://geoservices.ign.fr/sites/default/files/2021-07/SE_BDTOPO_avant_v3-0.pdf. Pas les mêmes classes de bâti, et ZONE_D_HABITATION était avant PAI_ZONE_HABITATION du thème I_ZONE_ACTIVITE.

Pas de gros changements avant ça. En Janvier 2009 changement de projections.

La classe BATIMENT de la version 1.2, très volumineuse (de l’ordre de 300 000 objets par département), a été divisée en plusieurs classes, de façon à faciliter et accélérer les applications tournant sur les bâtiments, en particulier les bâtiments possédant une fonction. En effet, les bâtiments indifférenciés représentent 90% de la totalité des bâtiments, les bâtiments dits remarquables environ 2% et les bâtiments à caractère industriel, agricole ou commercial 10%. Une classe CONSTRUCTION_LEGERE est créée. Les classes autres que « bâtiments » présents dans la version 1.2 se retrouvent dans la version 2.0 (réservoir, cimetière, terrain de sport, ...).

En plus d’une précision planimétrique et géométrique, les bâtiments ont une origine indiquant leur provenance (BDTopo, cadastre, terrain ou autre). Les surfaces et points d’activité ou d’intérêt (PAI) sont regroupés dans un thème à part, pour alléger le thème des bâtiments, d’autant plus qu’il existe de nouveaux types de PAI par rapport à la version 1.2 (voir paragraphe 1.4.2.6 Les zones et points d’activité ou d’intérêt).

## Annotations en général

There are many annotation formats, although PASCAL VOC and coco-json are the most commonly used. I recommend using geojson for storing polygons, then converting these to the required format when needed.

Longue list de softwares d'annotation : https://github.com/robmarkcole/satellite-image-deep-learning, ainsi que des outils pour visualiser les annotations et faire des conversions entre formats.

Des tas de tips et de bonnes pratiques ici aussi : https://github.com/robmarkcole/satellite-image-deep-learning

https://github.com/chrieke/awesome-satellite-imagery-datasets

- https://www.aicrowd.com/challenges/mapping-challenge