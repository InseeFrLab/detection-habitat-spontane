"""
Mappings.
"""


dep_to_crs = {
    "971": "4559",
    "973": "2972",
    "972": "4559",
    "976": "4471",
    "974": "2975",
    "977": "4559",
    "978": "4559",
}


name_dep_to_num_dep = {
    "GUADELOUPE": "971",
    "GUYANE": "973",
    "MARTINIQUE": "972",
    "MAYOTTE": "976",
    "REUNION": "974",
    "SAINT-BARTHELEMY": "977",
    "SAINT-MARTIN": "978",
}

num_dep_to_name_dep = {
    "971": "GUADELOUPE",
    "973": "GUYANE",
    "972": "MARTINIQUE",
    "976": "MAYOTTE",
    "974": "REUNION",
    "977": "SAINT-BARTHELEMY",
    "978": "SAINT-MARTIN",
}

name_dep_to_crs = {
    "GUADELOUPE": "4559",
    "MARTINIQUE": "4559",
    "MAYOTTE": "4471",
    "GUYANE": "2972",
    "REUNION": "2975",
    "SAINT-MARTIN": "4559",
}

name_dep_to_aoi = {
    "GUADELOUPE": {
        "west": -61.811124,
        "south": 15.828534,
        "east": -60.998518,
        "north": 16.523944,
    },
    "MARTINIQUE": {
        "west": -61.264617,
        "south": 14.378599,
        "east": -60.781573,
        "north": 14.899453,
    },
    "MAYOTTE": {
        "west": 45.013633,
        "south": -13.006619,
        "east": 45.308891,
        "north": -12.633022,
    },
    "GUYANE": {
        "west": -52.883,
        "south": 4.148,
        "east": -51.813,
        "north": 5.426,
    },
    "REUNION": {
        "west": 55.205,
        "south": -21.408,
        "east": 55.861,
        "north": -20.852,
    },
    "SAINT-MARTIN": {"west": -63.163, "south": 18.000, "east": -62.986614, "north": 18.129},
}
