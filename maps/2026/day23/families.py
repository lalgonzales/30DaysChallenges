import re

# Families follow the SCA/WCR Coffee Taster's Flavor Wheel (2016): its top
# tier, with Fruity split into its four second-tier groups. Cuppers wrote
# notes in English and Spanish, so both appear. Texture words (juicy, creamy,
# clean) are not flavours and get no family.
PHRASES = {  # checked first, as substrings
    "orange blossom": "Floral",
    "black tea": "Floral",
    "lemon tea": "Floral",
    "green tea": "Floral",
    "white tea": "Floral",
    "cherry blossom": "Floral",
    "brown sugar": "Sweet",
    "azucar morena": "Sweet",
    "graham cracker": "Sweet",
    "dried fruit": "Fruity: dried fruit",
    "dried dates": "Fruity: dried fruit",
    "coffee pulp": "Fruity: other fruit",
}
WORDS = [  # (family, keywords), first match wins
    (
        "Floral",
        "floral flower blossom jasmine jazmin rose chamomile lavender lavendar "
        "hibiscus roselle elderflower violet lilac honeysuckle geranium magnolia acacia tea tizana",
    ),
    (
        "Fruity: berry",
        "berry berries strawberry stawberry fresa raspberry rasberry blueberry "  # codespell:ignore stawberry,rasberry
        "blackberry mora cranberry currant boysenberry mulberry gooseberry bayas berrylike",
    ),
    ("Fruity: dried fruit", "raisin rasin prune date fig sultana pasa"),
    (
        "Fruity: citrus",
        "citrus citric citrico orange mandarin tangerine lemon lemonade lime "
        "limonario limon naranja grapefruit yuzu kumquat clementine pomelo bergamot",
    ),
    (
        "Fruity: other fruit",
        "fruit fruity fruta frutas apricot albaricoque peach melocoton durazno "
        "plum ciruela cherry cherries mango apple manzana pear grape grapes uva muscat pineapple "
        "papaya lychee melon honeydew cantaloupe watermelon passion nectarine guava kiwi banana "
        "tamarind coconut pomegranate persimmon starfruit carambola mangostine nance rhubarb quince jam jammy",
    ),
    ("Green & herbal", "herb herbal mint basil lemongrass tomato olive savory umami"),
    (
        "Fermented & winey",
        "wine winey rum whisky whiskey liquor fermented brandy cognac boozy "
        "port funky balsamic hops yogurt",
    ),
    (
        "Spices",
        "spice spices cinnamon canela clove nutmeg cardamom pepper anise ginger allspice",
    ),
    ("Roasted", "malt malta roast tobacco cereal toast"),
    (
        "Cocoa & nutty",
        "chocolate cocoa cacao nut nuts nutty almond hazelnut peanut peanuts "
        "walnut pecan macadamia praline",
    ),
    (
        "Sweet",
        "sweet caramel caramelized caramelo honey honeywater miel sugar sugarcane azucar "
        "molasses vanilla syrup maple toffee panela candy butterscotch marshmallow nougat cola",
    ),
]
WORDS = [(f, set(k.split())) for f, k in WORDS]


def family(term):
    term = term.lower().strip()
    for phrase, fam in PHRASES.items():
        if phrase in term:
            return fam
    words = set(re.findall(r"[a-z]+", term))
    words |= {w[:-1] for w in words if w.endswith("s")}
    for fam, keys in WORDS:
        if words & keys:
            return fam
    return None
