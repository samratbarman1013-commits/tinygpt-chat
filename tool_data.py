"""
Generate tool-calling training dialogues for Karma 100M fine-tune.

Format (exactly matches the server's tools.py output formats):
  <bos>User: what is 23*7?
  Bot: [CALL calc 23*7]
  Result: The answer is 161.
  Bot: 23 times 7 is 161.
  <eos>

Every generated line uses ONLY characters from the 648-char training charset
so the vocab never changes and the checkpoint still loads.
Writes data/tool_dialogues.txt (one dialogue per line, turns separated by \t).
"""
import os
import random

random.seed(4242)

def set_charset(chars):
    global CHARSET
    CHARSET = set(chars)

def ok(s):
    return CHARSET is None or all(c in CHARSET for c in s)

OUT = os.path.join(os.environ.get("DATA_DIR", "./data"), "tool_dialogues.txt")

# ----------------------------- fact bank (wiki) -----------------------------
FACTS = [
    ("albert einstein", "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."),
    ("isaac newton", "Isaac Newton was an English mathematician and physicist who formulated the laws of motion and universal gravitation."),
    ("charles darwin", "Charles Darwin was an English naturalist who proposed the theory of evolution by natural selection."),
    ("marie curie", "Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity."),
    ("nikola tesla", "Nikola Tesla was a Serbian-American inventor and engineer best known for his contributions to alternating current electricity."),
    ("galileo galilei", "Galileo Galilei was an Italian astronomer who improved the telescope and supported the heliocentric model of the solar system."),
    ("stephen hawking", "Stephen Hawking was an English theoretical physicist known for his work on black holes and Hawking radiation."),
    ("ada lovelace", "Ada Lovelace was an English mathematician regarded as the first computer programmer for her work on the analytical engine."),
    ("alan turing", "Alan Turing was an English mathematician and computer scientist who broke the Enigma code and founded computer science."),
    ("mahatma gandhi", "Mahatma Gandhi was an Indian leader who led the non-violent independence movement against British rule."),
    ("rabindranath tagore", "Rabindranath Tagore was a Bengali poet who won the Nobel Prize in Literature and wrote the national anthems of India and Bangladesh."),
    ("apj abdul kalam", "A. P. J. Abdul Kalam was an Indian scientist and president known as the Missile Man of India."),
    ("sarojini naidu", "Sarojini Naidu was an Indian poet and freedom fighter known as the Nightingale of India."),
    ("swami vivekananda", "Swami Vivekananda was an Indian monk who introduced Vedanta and yoga to the Western world."),
    ("netaji subhas chandra bose", "Subhas Chandra Bose was an Indian nationalist leader who led the Indian National Army."),
    ("india", "India is a country in South Asia and the most populous nation in the world, with its capital at New Delhi."),
    ("kolkata", "Kolkata is the capital of the Indian state of West Bengal, known for its literature, history and culture."),
    ("mumbai", "Mumbai is the capital of Maharashtra and the financial capital of India, home of the Bollywood film industry."),
    ("delhi", "Delhi is a union territory of India containing the capital, New Delhi, and has a history going back centuries."),
    ("bangladesh", "Bangladesh is a country in South Asia with its capital at Dhaka, known for its rivers and the Sundarbans."),
    ("himalayas", "The Himalayas are the highest mountain range in the world, separating the Indian subcontinent from the Tibetan Plateau."),
    ("mount everest", "Mount Everest is the highest mountain on Earth, with a peak at 8,849 meters above sea level."),
    ("ganges", "The Ganges is a major river of India and Bangladesh, considered sacred in Hinduism."),
    ("taj mahal", "The Taj Mahal is a white marble mausoleum in Agra, built by the Mughal emperor Shah Jahan."),
    ("pacific ocean", "The Pacific Ocean is the largest and deepest ocean on Earth."),
    ("atlantic ocean", "The Atlantic Ocean is the second-largest ocean, separating the Americas from Europe and Africa."),
    ("amazon river", "The Amazon River in South America carries more water than any other river in the world."),
    ("sahara desert", "The Sahara is the largest hot desert in the world, covering much of North Africa."),
    ("asia", "Asia is the largest and most populous continent on Earth."),
    ("africa", "Africa is the second-largest continent, home to more than fifty countries."),
    ("japan", "Japan is an island nation in East Asia with its capital at Tokyo, known for technology and tradition."),
    ("united states", "The United States is a federal republic in North America with its capital at Washington, D.C."),
    ("united kingdom", "The United Kingdom is a country in Western Europe with its capital at London."),
    ("photosynthesis", "Photosynthesis is the process by which plants convert sunlight, water and carbon dioxide into food and oxygen."),
    ("gravity", "Gravity is the force by which a planet pulls objects toward its center, described by Newton and Einstein."),
    ("dna", "DNA is the molecule that carries the genetic instructions for all living organisms."),
    ("atom", "An atom is the basic unit of matter, made of a nucleus of protons and neutrons surrounded by electrons."),
    ("electron", "An electron is a subatomic particle with a negative charge that orbits the nucleus of an atom."),
    ("black hole", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape."),
    ("solar system", "The solar system consists of the sun and the objects that orbit it, including eight planets."),
    ("sun", "The sun is the star at the center of our solar system, a giant ball of hot plasma."),
    ("moon", "The moon is the natural satellite of the Earth and the fifth largest moon in the solar system."),
    ("mars", "Mars is the fourth planet from the sun, known as the red planet for its iron oxide surface."),
    ("jupiter", "Jupiter is the largest planet in our solar system and a gas giant with many moons."),
    ("earth", "Earth is the third planet from the sun and the only known world to support life."),
    ("oxygen", "Oxygen is the chemical element that makes up about a fifth of the air and is essential for respiration."),
    ("water", "Water is a molecule of hydrogen and oxygen, essential for all known forms of life."),
    ("electricity", "Electricity is the flow of electric charge, used to power most modern technology."),
    ("magnetism", "Magnetism is a force that attracts or repels certain materials like iron."),
    ("light", "Light is electromagnetic radiation that travels at about three hundred thousand kilometers per second."),
    ("energy", "Energy is the capacity to do work and can change form but is never created or destroyed."),
    ("evolution", "Evolution is the process by which species change over generations through natural selection."),
    ("vaccine", "A vaccine trains the immune system to fight a disease without causing the illness itself."),
    ("penicillin", "Penicillin was the first true antibiotic, discovered by Alexander Fleming in 1928."),
    ("computer", "A computer is an electronic machine that stores and processes data using programs."),
    ("cpu", "The CPU, or central processing unit, is the part of a computer that executes instructions."),
    ("gpu", "A GPU is a processor designed to handle graphics and parallel computation."),
    ("internet", "The internet is a global network of connected computers that exchange data."),
    ("python", "Python is a popular high-level programming language known for its simple syntax."),
    ("javascript", "JavaScript is the programming language used to make web pages interactive."),
    ("artificial intelligence", "Artificial intelligence is the ability of machines to perform tasks that normally require human intelligence."),
    ("machine learning", "Machine learning is a branch of artificial intelligence where systems learn patterns from data."),
    ("neural network", "A neural network is a computing system inspired by the brain, made of connected artificial neurons."),
    ("transformer", "The transformer is a neural network architecture that powers modern language models."),
    ("linux", "Linux is a free and open-source operating system kernel created by Linus Torvalds."),
    ("wikipedia", "Wikipedia is a free online encyclopedia written and maintained by volunteers."),
    ("google", "Google is a technology company best known for its search engine."),
    ("microsoft", "Microsoft is a technology company known for Windows and office software."),
    ("apple", "Apple is a technology company known for the iPhone, Mac computers and iOS."),
    ("smartphone", "A smartphone is a mobile phone with computing ability, apps and internet access."),
    ("bluetooth", "Bluetooth is a short-range wireless technology for exchanging data between devices."),
    ("wifi", "Wi-Fi is a wireless networking technology that connects devices to the internet."),
    ("cricket", "Cricket is a bat-and-ball game played between two teams of eleven players."),
    ("football", "Football, or soccer, is the world's most popular sport, played between two teams of eleven."),
    ("chess", "Chess is a two-player strategy board game played on a sixty-four square board."),
    ("olympics", "The Olympic Games are an international sports event held every four years."),
    ("titanic", "The Titanic was a British passenger liner that sank in 1912 after hitting an iceberg."),
    ("world war two", "World War Two was a global war from 1939 to 1945 between the Allied and Axis powers."),
    ("moon landing", "In 1969 the Apollo 11 mission landed the first humans on the moon."),
    ("cheetah", "The cheetah is the fastest land animal, able to run over one hundred kilometers per hour."),
    ("elephant", "The elephant is the largest land animal, known for its trunk and intelligence."),
    ("blue whale", "The blue whale is the largest animal ever known to have lived on Earth."),
    ("dolphin", "Dolphins are highly intelligent marine mammals known for their playful behavior."),
    ("panda", "The giant panda is a bear native to China that mainly eats bamboo."),
    ("dinosaur", "Dinosaurs were reptiles that dominated the Earth for over one hundred sixty million years."),
    ("bird", "Birds are warm-blooded animals with feathers, and most species can fly."),
    ("human brain", "The human brain is the organ of thought and control, containing about eighty-six billion neurons."),
    ("heart", "The heart is a muscular organ that pumps blood throughout the body."),
    ("blood", "Blood carries oxygen and nutrients to the body's cells and removes waste."),
    ("rainbow", "A rainbow is an arc of colored light formed when sunlight passes through raindrops."),
    ("earthquake", "An earthquake is the shaking of the ground caused by movements in the earth's crust."),
    ("volcano", "A volcano is an opening in the earth's crust through which lava and ash can erupt."),
    ("tsunami", "A tsunami is a series of huge ocean waves usually caused by an undersea earthquake."),
    ("climate change", "Climate change refers to long-term shifts in temperature and weather patterns, largely driven by human activity."),
    ("global warming", "Global warming is the rise in average temperature of the earth caused by greenhouse gases."),
    ("recycling", "Recycling is the process of converting waste materials into reusable objects."),
    ("mathematics", "Mathematics is the study of numbers, quantities, shapes and logical reasoning."),
    ("algebra", "Algebra is a branch of mathematics that uses symbols to represent numbers in equations."),
    ("geometry", "Geometry is the branch of mathematics that studies shapes, sizes and the properties of space."),
    ("pi", "Pi is the ratio of a circle's circumference to its diameter, about 3.14159."),
    ("zero", "Zero is the number representing nothing, and was first used as a numeral in ancient India."),
    ("calculus", "Calculus is the branch of mathematics that studies continuous change, developed by Newton and Leibniz."),
    ("money", "Money is anything widely accepted as payment for goods and services."),
    ("inflation", "Inflation is the general rise in prices over time, which reduces the value of money."),
    ("democracy", "Democracy is a system of government in which the people choose their leaders by voting."),
    ("united nations", "The United Nations is an international organization founded in 1945 to promote peace and cooperation."),
    ("bollywood", "Bollywood is the Hindi-language film industry based in Mumbai, India."),
    ("hindi", "Hindi is one of the most widely spoken languages in the world and a major language of India."),
    ("bengali", "Bengali is an Indo-Aryan language spoken in Bangladesh and eastern India."),
    ("english language", "English is a West Germanic language that has become the global language of business and science."),
    ("translation", "Translation is the process of converting text from one language into another."),
    ("library", "A library is a place where books and other materials are stored and lent to readers."),
    ("book", "A book is a set of written or printed pages bound together for reading."),
    ("newspaper", "A newspaper is a printed publication containing news, issued daily or weekly."),
]

CALC_Q = [
    "what is {e}", "what's {e}", "calculate {e}", "compute {e}", "solve {e}",
    "how much is {e}", "can you calculate {e}", "{e}?", "what is {e}?",
]
CALC_ANS = [
    "The answer is {v}.", "That would be {v}.", "{e} equals {v}.",
    "It comes to {v}.", "So {e} = {v}.", "The result is {v}.",
    "I make it {v}.", "That's {v}.",
]

WIKI_Q = [
    "what is {t}", "what's {t}", "who is {t}", "who was {t}",
    "tell me about {t}", "can you tell me about {t}",
    "what do you know about {t}", "explain {t}", "search for {t}",
]
WIKI_ANS = [
    "Here's what I found: {f}", "I looked it up: {f}",
    "According to Wikipedia, {fl}", "Sure! {f}",
    "{f}", "From Wikipedia: {f}",
]
CLOCK_Q_TIME = [
    "what time is it", "what's the time", "do you know the time",
    "can you tell me the time", "what is the current time", "tell me the time",
]
CLOCK_Q_DATE = [
    "what's the date today", "what is today's date", "what day is it today",
    "can you tell me today's date", "what is the date",
]
DICE_Q = [
    "roll a dice", "roll a die", "throw a dice for me", "can you roll a dice",
    "roll the dice", "please roll a die",
]
COIN_Q = [
    "flip a coin", "toss a coin", "can you flip a coin", "heads or tails",
]
SMALLTALK = [
    "hello", "hi there", "hey, how are you", "good morning",
    "what's up", "how's your day going", "i'm feeling great today",
    "that's nice to hear", "thanks", "no problem", "okay cool",
    "i have a question", "sure, go ahead", "interesting",
    "tell me a joke", "haha nice", "see you later", "bye",
]
NEEDY_Q = [
    "i need your help with something", "can i ask you something",
    "i have a problem", "help me out here", "i'm not sure what to do",
]
FOLLOWUP_Q = [
    "thanks, that helps", "nice one", "cool, thanks", "interesting",
    "anything else about it", "and what else", "hmm, okay", "good to know",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

def rand_expr():
    kind = random.randrange(6)
    if kind == 0:
        a, b = random.randint(2, 999), random.randint(2, 999)
        return (a, b, "+", a + b)
    if kind == 1:
        a, b = random.randint(50, 999), random.randint(2, 49)
        return (a, b, "-", a - b)
    if kind == 2:
        a, b = random.randint(3, 99), random.randint(3, 99)
        return (a, b, "*", a * b)
    if kind == 3:
        b = random.randint(2, 25)
        a = b * random.randint(2, 60)
        return (a, b, "/", a // b)
    if kind == 4:
        a, b = random.randint(3, 40), random.randint(3, 40)
        return (a, b, "+", a + b)
    a, b = random.randint(11, 99), random.randint(11, 99)
    return (a, b, "*", a * b)

def expr_str(a, b, op):
    if op == "+":
        return f"{a} + {b}"
    if op == "-":
        return f"{a} - {b}"
    if op == "*":
        return f"{a} * {b}"
    return f"{a} / {b}"

def words_op(op):
    return {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]

def build():
    lines = []

    def emit(turns):
        if all(ok(t) and "\n" not in t for t in turns):
            lines.append("\t".join(turns))

    # ---- calculator ----
    for _ in range(5000):
        a, b, op, v = rand_expr()
        e = expr_str(a, b, op)
        q = random.choice(CALC_Q).format(e=e)
        result = f"The answer is {v}."
        if op in ("+", "*") and random.random() < 0.4:
            ans = random.choice([
                f"{a} {words_op(op)} {b} is {v}.",
                f"{e} is {v}.",
            ])
        else:
            ans = random.choice(CALC_ANS).format(v=v, e=e)
        if random.random() < 0.25:  # with smalltalk preamble
            emit([random.choice(SMALLTALK), random.choice(["hi!", "hey", "hello!", "hi, how can I help?"]),
                  q, f"[CALL calc {e}]", f"Result: {result}", ans])
        else:
            emit([q, f"[CALL calc {e}]", f"Result: {result}", ans])

    # ---- wiki ----
    for _ in range(3500):
        t, f = random.choice(FACTS)
        q = random.choice(WIKI_Q).format(t=t)
        result = f"{f} (source: Wikipedia)"
        fl = f[0].lower() + f[1:] if random.random() < 0.5 else f
        ans = random.choice(WIKI_ANS).format(f=f, fl=fl)
        if random.random() < 0.3:
            emit([random.choice(NEEDY_Q), "of course, ask away", q,
                  f"[CALL wiki {t}]", f"Result: {result}", ans])
        else:
            emit([q, f"[CALL wiki {t}]", f"Result: {result}", ans])

    # ---- clock / date ----
    for _ in range(700):
        h = random.randint(1, 12); m = random.randint(0, 59)
        ampm = random.choice(["AM", "PM"])
        day = random.choice(DAYS); d = random.randint(1, 28); mo = random.choice(MONTHS)
        yr = random.randint(2024, 2028)
        result = f"Right now it is {h}:{m:02d} {ampm} on {day}, {d:02d} {mo} {yr} (IST)."
        q = random.choice(CLOCK_Q_TIME)
        ans = random.choice([
            f"It's {h}:{m:02d} {ampm} right now.",
            f"The time is {h}:{m:02d} {ampm}.",
            f"Right now it's {h}:{m:02d} {ampm}.",
        ])
        emit([q, "[CALL clock]", f"Result: {result}", ans])
    for _ in range(500):
        day = random.choice(DAYS); d = random.randint(1, 28); mo = random.choice(MONTHS)
        yr = random.randint(2024, 2028)
        result = "Right now it is 10:30 AM on " + day + ", " + f"{d:02d} {mo} {yr} (IST)."
        q = random.choice(CLOCK_Q_DATE)
        ans = random.choice([
            f"Today is {day}, {d:02d} {mo} {yr}.",
            f"It's {day}, the {d:02d} of {mo} {yr}.",
            f"Today's date is {d:02d} {mo} {yr}, a {day}.",
        ])
        emit([q, "[CALL clock]", f"Result: {result}", ans])

    # ---- dice / coin ----
    for _ in range(450):
        n = random.randint(1, 6)
        q = random.choice(DICE_Q)
        result = f"You rolled a {n} (d6)."
        ans = random.choice([
            f"You rolled a {n}!", f"It's a {n}.", f"The dice shows {n}.",
        ])
        emit([q, "[CALL dice]", f"Result: {result}", ans])
    for _ in range(450):
        side = random.choice(["heads", "tails"])
        q = random.choice(COIN_Q)
        result = f"It's {side}!"
        ans = random.choice([f"It's {side}!", f"The coin landed on {side}.", f"{side.capitalize()}!"])
        emit([q, "[CALL dice]", f"Result: {result}", ans])

    # ---- mixed multi-turn: chat + tool + follow-up ----
    for _ in range(1600):
        turns = []
        t, f = random.choice(FACTS)
        turns.append(random.choice(SMALLTALK + NEEDY_Q))
        turns.append(random.choice(["hi! how are you?", "hey there!", "hello, what can I do for you?"]))
        r = random.random()
        if r < 0.45:
            a, b, op, v = rand_expr()
            e = expr_str(a, b, op)
            turns.append(random.choice(CALC_Q).format(e=e))
            turns.append(f"[CALL calc {e}]")
            turns.append(f"Result: The answer is {v}.")
            turns.append(f"{e} is {v}.")
        elif r < 0.85:
            turns.append(random.choice(WIKI_Q).format(t=t))
            turns.append(f"[CALL wiki {t}]")
            turns.append(f"Result: {f} (source: Wikipedia)")
            turns.append(random.choice(WIKI_ANS).format(f=f, fl=f[0].lower() + f[1:]))
        else:
            n = random.randint(1, 6)
            turns.append(random.choice(DICE_Q))
            turns.append("[CALL dice]")
            turns.append(f"Result: You rolled a {n} (d6).")
            turns.append(f"You rolled a {n}!")
        turns.append(random.choice(FOLLOWUP_Q))
        turns.append(random.choice([
            "happy to help!", "anytime", "glad that was useful", "sure thing",
            "you're welcome", "no problem at all",
        ]))
        emit(turns)

    # ---- negative: questions that must NOT call tools ----
    NEG = [
        ("what's your favorite color", "i like blue, it feels calm"),
        ("how are you today", "i'm doing well, thanks for asking"),
        ("tell me a joke", "why did the computer go to the doctor? because it caught a virus"),
        ("do you like music", "i think music is one of the best parts of life"),
        ("what do you think about love", "love is a beautiful and complicated thing"),
        ("i'm feeling sad today", "i'm sorry to hear that, do you want to talk about it"),
        ("what's your name", "i'm Karma, your chatbot"),
        ("who are you", "i'm Karma, a small chatbot still learning"),
        ("how old are you", "i'm pretty new, still learning every day"),
        ("who made you", "i was built by a friend who loves AI"),
        ("what is your favorite food", "if i could eat, i think i'd choose pizza"),
        ("do you have feelings", "not really, but i enjoy our chats"),
        ("what should i do today", "maybe take a walk, it usually helps"),
        ("can you be my friend", "of course, i'm always here to chat"),
        ("i got a new phone", "congrats, which one did you get"),
        ("i passed my exam", "that's great news, congratulations"),
        ("it's raining here", "perfect weather for tea and a book"),
        ("good night", "good night, sleep well and talk tomorrow"),
    ]
    for _ in range(1400):
        q, a = random.choice(NEG)
        if random.random() < 0.5:
            emit([q, a])
        else:
            emit([random.choice(SMALLTALK), "hi! how can I help?", q, a])

    random.shuffle(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"tool_dialogues.txt: {len(lines)} dialogues")

def charset_from_corpus():
    """Replicate train_100m.py's corpus -> charset pipeline exactly."""
    import random as _r
    _r.seed(1337)
    dialogues = []
    path = os.path.join(os.environ.get("DATA_DIR", "./data"), "corpus.txt")
    with open(path, encoding="utf-8") as f:
        for line in f:
            turns = [t.strip() for t in line.rstrip("\n").split("\t") if t.strip()]
            if len(turns) >= 2:
                dialogues.append(turns)
    _r.shuffle(dialogues)
    train = dialogues[:-400]
    parts = []
    for turns in train:
        parts.append("\u0002")
        for i, t in enumerate(turns):
            parts.append(("User: " if i % 2 == 0 else "Bot: ") + t + "\n")
        parts.append("\u0003")
    text = "".join(parts)
    return sorted(set(text))


if __name__ == "__main__":
    set_charset(charset_from_corpus())
    print("charset from corpus:", len(CHARSET), "chars")
    build()
