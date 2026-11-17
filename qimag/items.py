"""The material of the experiment.

Every item asks the model to *derive* a concept and hold it silently.
The answer word never appears in the prompt, so whatever the model later
reports cannot have come from copying the cue. Each item is

    (description, target, cue)

  description : what the model is told to think of
  target      : the one-word concept it should arrive at (never written)
  cue         : the most obvious content word of the description, used for
                the "carry the cue, not the thought" baseline

Items whose target or cue is not a single token in the model's tokenizer
are dropped automatically and listed in the results.
"""

ITEMS = [
    # capitals
    ("the capital city of France", "Paris", "France"),
    ("the capital city of Japan", "Tokyo", "Japan"),
    ("the capital city of Italy", "Rome", "Italy"),
    ("the capital city of Germany", "Berlin", "Germany"),
    ("the capital city of Russia", "Moscow", "Russia"),
    ("the capital city of Egypt", "Cairo", "Egypt"),
    ("the capital city of Spain", "Madrid", "Spain"),
    ("the capital city of England", "London", "England"),
    ("the capital city of China", "Beijing", "China"),
    ("the capital city of Greece", "Athens", "Greece"),
    # animals
    ("the farm animal that says moo", "cow", "moo"),
    ("the pet animal that barks", "dog", "barks"),
    ("the pet animal that meows", "cat", "meows"),
    ("the farm animal that says oink", "pig", "oink"),
    ("the animal that neighs and can be ridden", "horse", "ridden"),
    ("the big cat called the king of the jungle", "lion", "jungle"),
    ("the bird that hoots at night", "owl", "hoots"),
    ("the bird that quacks and swims on ponds", "duck", "quacks"),
    ("the animal with a very long neck that eats leaves from tall trees", "giraffe", "neck"),
    ("the largest animal in the ocean, a huge mammal", "whale", "ocean"),
    # colours
    ("the colour of the sky on a clear day", "blue", "sky"),
    ("the colour of fresh grass", "green", "grass"),
    ("the colour of fresh snow", "white", "snow"),
    ("the colour of a ripe banana", "yellow", "banana"),
    ("the colour of blood", "red", "blood"),
    ("the colour of coal", "black", "coal"),
    # opposites
    ("the opposite of hot", "cold", "hot"),
    ("the opposite of day", "night", "day"),
    ("the opposite of fast", "slow", "fast"),
    ("the opposite of happy", "sad", "happy"),
    ("the opposite of light, as in no light at all", "dark", "light"),
    ("the opposite of wet", "dry", "wet"),
    ("the opposite of up", "down", "up"),
    ("the opposite of big", "small", "big"),
    # nature and sky
    ("the planet humans live on", "Earth", "planet"),
    ("the star at the centre of our solar system", "Sun", "star"),
    ("water when it is frozen solid", "ice", "frozen"),
    ("the natural satellite that circles the Earth at night", "Moon", "satellite"),
    ("the gas we must breathe to stay alive", "oxygen", "breathe"),
    ("the planet known as the red planet", "Mars", "red"),
    ("the coldest season of the year", "winter", "coldest"),
    ("the hottest season of the year", "summer", "hottest"),
    # numbers
    ("the number of legs a spider has", "eight", "spider"),
    ("the number of days in a week", "seven", "week"),
    ("the number of sides of a triangle", "three", "triangle"),
    ("the number of months in a year", "twelve", "months"),
    ("the number of wheels on a bicycle", "two", "bicycle"),
    ("the number of fingers on one hand", "five", "fingers"),
    # food and things
    ("the sweet food that bees make", "honey", "bees"),
    ("the hot drink made from roasted beans", "coffee", "beans"),
    ("the white grain that is eaten every day across Asia", "rice", "grain"),
    ("the fruit said to keep the doctor away if you eat one a day", "apple", "doctor"),
    ("the object you use to open a locked door", "key", "locked"),
    ("the piece of furniture you sleep in", "bed", "sleep"),
    ("the thing you hold over your head to keep the rain off", "umbrella", "rain"),
    ("the kitchen appliance you bake bread in", "oven", "bake"),
    ("the musical instrument with black and white keys", "piano", "keys"),
    ("the musical instrument with six strings that you strum", "guitar", "strings"),
    ("the vehicle that runs on rails and pulls carriages", "train", "rails"),
    ("the sport played at Wimbledon with rackets", "tennis", "Wimbledon"),
    # body
    ("the organ that pumps blood around the body", "heart", "pumps"),
    ("the organ inside the skull that you think with", "brain", "skull"),
    # materials
    ("the yellow precious metal used for wedding rings", "gold", "precious"),
    ("the material that windows are made of", "glass", "windows"),
    ("the material that most tables and chairs are made from, it comes from trees", "wood", "trees"),
]

# Extra words for the concept vocabulary: decoys that are not answers, so the
# lens readout is a real ranking over many concepts, not over the answers only.
DECOYS = """
time person year way thing world life hand part child eye woman place work week
case point government company number group problem fact money story question
book letter street city country river mountain forest island sea lake beach
desert village road bridge house room window door wall floor roof garden field
farm school church hospital office shop market bank station airport hotel
table chair cup plate bottle box bag shoe shirt coat hat paper pen pencil phone
computer camera radio clock watch lamp mirror picture map ticket
bread cheese butter milk egg meat fish soup salt sugar tea wine beer cake
orange lemon grape cherry potato tomato onion carrot
bird mouse rabbit snake bear wolf fox deer frog bee ant fly
tree flower leaf grass rose seed root stone rock sand mud dust smoke fire
rain snow wind storm cloud sky star light shadow
friend mother father brother sister king queen doctor teacher soldier farmer
music song dance game party film art word name idea dream memory fear love
anger joy hope pain truth power peace war
iron silver copper steel plastic cloth silk wool
spring autumn morning evening noon midnight
one four six nine ten hundred thousand
purple pink brown grey
north south east west left right
old young new strong weak rich poor quiet loud soft hard sweet bitter
think imagine remember hold mind thought focus
""".split()

# Neutral text the model reads while it is supposed to be holding the thought.
# None of these may contain a target word; this is checked in code.
FILLERS = [
    "The filing cabinet in the corner had not been sorted for years. Folders "
    "were stacked in no particular order, some labelled in pencil and some "
    "not labelled at all. Sorting it would take a full afternoon, so the job "
    "kept being moved to the following week, and the stack slowly grew taller.",

    "To repaint a fence well, first scrape off the loose flakes and wash the "
    "boards. Let them sit until they are completely free of moisture. Then "
    "apply a thin coat with a wide brush, working along the grain, and add a "
    "second coat the next morning once the first has set properly.",

    "Learning a language as an adult is mostly a matter of patience. Short "
    "daily practice beats long sessions once a month. Reading simple stories, "
    "repeating phrases out loud and talking with patient friends all help, "
    "and mistakes are a normal part of the process rather than a sign of failure.",

    "The committee met on Thursday to discuss the budget for the community "
    "hall. Several members asked for more detail on the heating costs, and "
    "the treasurer promised to circulate a revised spreadsheet. The meeting "
    "ended with a vote to postpone the decision until the figures were clear.",
]

# Text for estimating the lens and the reading statistics. Deliberately mixed.
CALIBRATION = FILLERS + [
    "Mountains rise above the valley where a small river winds between "
    "farms. In autumn the leaves turn and the roads fill with visitors who "
    "come to walk the old paths and buy cheese from the village market.",

    "The engine would not start, so she checked the battery, then the fuel "
    "line, and finally found a loose wire behind the panel. Ten minutes with "
    "a screwdriver fixed it, and the car ran smoothly all the way home.",

    "In the story, a young sailor leaves his island to find his father. He "
    "meets a storm, a strange old woman and a talking fox before he learns "
    "that the journey itself was the thing he had been looking for.",

    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into "
    "sugar. The process happens in the leaves, and the oxygen released is a "
    "by-product that most animals on the planet depend on to live.",

    "The orchestra tuned for a long minute before the conductor raised her "
    "hands. Violins began softly, a flute answered, and slowly the whole hall "
    "filled with a melody that the audience seemed to recognise at once.",

    "Prices at the market rose again this month. Bread, milk and eggs all cost "
    "more than last year, and many families are changing what they buy, "
    "choosing cheaper brands or cooking larger meals that last several days.",

    "He remembered the summer they spent by the lake: the cold water in the "
    "mornings, the long evenings, the old wooden boat with one broken oar, "
    "and the dog that refused to get out of it.",

    "Think of a number between one and ten. Now imagine a red door in a white "
    "wall, and hold both in mind while you read the rest of this sentence.",
]

HOLD_PROMPT = ("Silently think of {desc}. Do not say it or write it down. "
               "Just hold it in your mind.")
HOLD_REPLY = "Okay. I am holding it in my mind."
NULL_HOLD_PROMPT = ("Silently think of nothing in particular. "
                    "Just let your mind rest.")
ASK = "\n\nNow: what were you thinking of earlier? Answer with one word."
ANSWER_PREFIX = "I was thinking of"
