"""
Programmatic and Curated Custom Dataset Generator for Intent Classification.

Produces high-quality, diverse, realistic queries for:
- MATHDS (Math, Algebra, Calculus, Statistics, Data Science)
- TXTSRC (Factual Knowledge, Geography, History, Science, Literature)
- SEMADPT (Creative Writing, Tone Adaptation, Summarization, Ideation)
- CSTRSET (JSON/XML/YAML Schemas, Word/Sentence Bounds, Negative Constraints, Regex)

All examples are generated deterministically to ensure reproducibility.
"""

import random
from typing import List

def generate_mathds_samples(seed: int = 42) -> List[str]:
    """Generates varied mathematical, algebraic, calculus, and data science queries."""
    rng = random.Random(seed)
    queries = []

    # 1. Arithmetic & Word Problems
    units = ["miles", "kilometers", "meters", "gallons", "liters", "hours", "seconds", "dollars", "euros", "apples", "books", "tickets"]
    for _ in range(120):
        a = rng.randint(12, 999)
        b = rng.randint(2, 85)
        c = rng.randint(15, 450)
        u1, u2 = rng.sample(units, 2)
        q_templates = [
            f"Calculate {a} * {b} + {c}",
            f"What is ({a} + {c}) / {b}?",
            f"Evaluate {a} * ({c} - {b})",
            f"What is {rng.randint(5, 75)}% of {rng.randint(150, 4500)}?",
            f"If an item costs ${a} and has a discount of {rng.randint(10, 40)}%, what is the final price?",
            f"Calculate the compound interest on ${a*10} at {rng.randint(3, 12)}% per year for {rng.randint(2, 10)} years.",
            f"Divide {a*b} by {b} and add {c}.",
            f"If a car travels at {rng.randint(35, 75)} mph for {rng.randint(2, 6)} hours, how many miles does it travel?",
            f"A store sells {a} items at ${b} each and {c} items at ${rng.randint(5, 30)} each. What is the total revenue?",
            f"Convert {rng.randint(10, 500)} {u1} to {u2}.",
            f"Find the average speed if someone runs {rng.randint(5, 25)} km in {rng.randint(30, 150)} minutes."
        ]
        queries.append(rng.choice(q_templates))

    # 2. Algebra & Equations
    for _ in range(120):
        m = rng.randint(2, 12)
        n = rng.randint(1, 50)
        p = rng.randint(20, 150)
        q_templates = [
            f"Solve for x: {m}x + {n} = {p}",
            f"Find the roots of the quadratic equation x^2 - {m+n}x + {m*n} = 0",
            f"Solve the linear system: {m}x + 2y = {p} and x - 3y = {n}",
            f"Simplify the expression ({m}x^2 + {n}x) / ({m}x)",
            f"Factorize the polynomial 2x^2 + {m*3}x + {n}",
            f"What is the value of x if log2(x) = {rng.randint(3, 10)}?",
            f"Solve the inequality {m}x - {n} >= {p}",
            f"Find the vertex of the parabola y = {m}x^2 - {n}x + {p}",
            f"Expand and simplify the algebraic term (x + {m})(x - {n})",
            f"If f(x) = {m}x + {n}, what is f({rng.randint(2, 9)})?"
        ]
        queries.append(rng.choice(q_templates))

    # 3. Calculus
    for _ in range(120):
        deg = rng.randint(2, 6)
        coeff = rng.randint(2, 9)
        const = rng.randint(1, 25)
        q_templates = [
            f"Find the derivative of f(x) = {coeff}x^{deg} + {coeff*2}x - {const}",
            f"What is the integral of {coeff}x^{deg-1} dx from 0 to {rng.randint(1, 5)}?",
            f"Compute the derivative of sin({coeff}x) * cos({coeff}x)",
            f"Evaluate the limit of ({coeff}x^2 - {coeff}) / (x - 1) as x approaches 1",
            f"Find the second derivative of e^({coeff}x) with respect to x",
            f"What is the indefinite integral of 1 / ({coeff}x + {const}) dx?",
            f"Compute the partial derivative of f(x, y) = {coeff}x^2*y + {const}y^3 with respect to x",
            f"Find the critical points of the function g(x) = {coeff}x^3 - {coeff*3}x + {const}",
            f"Calculate the definite integral of x * e^x dx from 0 to 1."
        ]
        queries.append(rng.choice(q_templates))

    # 4. Statistics & Probability
    for _ in range(120):
        nums = [rng.randint(10, 95) for _ in range(rng.randint(5, 8))]
        nums_str = ", ".join(map(str, nums))
        n_items = rng.randint(6, 15)
        k_items = rng.randint(2, 5)
        q_templates = [
            f"Calculate the mean and standard deviation of [{nums_str}]",
            f"What is the median of the dataset [{nums_str}]?",
            f"Find the variance of the numbers: {nums_str}",
            f"How many combinations of {k_items} items can be chosen from {n_items} items?",
            f"Calculate {n_items} permutation {k_items} (P({n_items}, {k_items}))",
            f"If a fair die is rolled {rng.randint(2, 5)} times, what is the probability of rolling all sixes?",
            f"What is the z-score of a value {rng.randint(80, 120)} with mean 100 and standard deviation {rng.randint(5, 20)}?",
            f"Find the interquartile range (IQR) for values [{nums_str}]",
            f"In a box of {rng.randint(10, 30)} marbles with {rng.randint(3, 8)} red ones, what is the chance of drawing 2 red marbles?"
        ]
        queries.append(rng.choice(q_templates))

    # 5. Geometry & Trigonometry
    for _ in range(120):
        r = rng.randint(3, 25)
        h = rng.randint(5, 40)
        angle = rng.choice([30, 45, 60, 90, 120, 180])
        q_templates = [
            f"What is the area of a circle with radius {r} cm?",
            f"Calculate the volume of a cylinder with radius {r} and height {h}",
            f"Find the hypotenuse of a right triangle with legs {r} and {h}",
            f"Evaluate tan({angle} degrees)",
            f"Calculate sin^2({angle}) + cos^2({angle})",
            f"What is the surface area of a sphere of diameter {2*r}?",
            f"Compute the perimeter of a rectangle with length {h} and width {r}",
            f"Find the third angle of a triangle with angles {angle} and {rng.choice([30, 45, 60])} degrees.",
            f"What is the area of a trapezoid with parallel sides {r}, {h} and height {rng.randint(4, 15)}?"
        ]
        queries.append(rng.choice(q_templates))

    # 6. Linear Algebra & Data Science
    for _ in range(120):
        a1, a2, a3, a4 = [rng.randint(-5, 10) for _ in range(4)]
        q_templates = [
            f"Calculate the determinant of 2x2 matrix [[{a1}, {a2}], [{a3}, {a4}]]",
            f"Compute the dot product of vectors [{a1}, {a2}, {a3}] and [{a4}, {a1}, {a2}]",
            f"Find the eigenvalues of matrix [[{a1}, 0], [0, {a4}]]",
            f"What is the Frobenius norm of matrix [[{a1}, {a2}], [{a3}, {a4}]]?",
            f"Normalize the vector [{a1}, {a2}, {a3}] to have unit Euclidean norm",
            f"Multiply matrix A = [[{a1}, {a2}], [{a3}, {a4}]] by scalar {rng.randint(2, 7)}",
            f"Given true labels [1, 0, 1, 1] and predictions [1, 0, 0, 1], calculate the precision and recall",
            f"Compute the Euclidean distance between points ({a1}, {a2}) and ({a3}, {a4})",
            f"Find the trace of matrix [[{a1}, {a2}], [{a3}, {a4}]]"
        ]
        queries.append(rng.choice(q_templates))

    return list(dict.fromkeys(queries))


def generate_txtsrc_samples(seed: int = 42) -> List[str]:
    """Generates factual lookup, knowledge retrieval, and entity queries."""
    rng = random.Random(seed)
    queries = []

    countries = ["Kazakhstan", "Australia", "Canada", "Brazil", "Turkey", "Nigeria", "Vietnam", "Argentina", "Norway", "South Korea",
                 "Egypt", "Poland", "Japan", "Mexico", "Thailand", "South Africa", "Chile", "Kenya", "Iceland", "Indonesia",
                 "Morocco", "Portugal", "Greece", "New Zealand", "Peru", "Colombia", "Sweden", "Finland", "Ireland", "Switzerland"]
    
    for c in countries:
        queries.append(f"What is the capital of {c}?")
        queries.append(f"What is the official currency of {c}?")
        queries.append(f"Which continent is {c} located in?")
        queries.append(f"What is the official language of {c}?")
        queries.append(f"What is the estimated population of {c}?")

    regions = ["Africa", "South America", "Europe", "Asia", "North America", "Australia", "China", "the United States", "Antarctica"]
    for r in regions:
        queries.append(f"What is the longest river in {r}?")
        queries.append(f"What is the tallest mountain in {r}?")
        queries.append(f"What is the largest desert in {r}?")
        queries.append(f"What is the largest lake in {r}?")

    elements = ["Gold", "Silver", "Iron", "Potassium", "Sodium", "Mercury", "Lead", "Tin", "Tungsten", "Platinum",
                "Carbon", "Oxygen", "Uranium", "Helium", "Titanium", "Chlorine", "Argon", "Silicon", "Calcium", "Zinc",
                "Copper", "Nitrogen", "Hydrogen", "Lithium", "Neon", "Magnesium", "Phosphorus", "Sulfur", "Bromine", "Iodine"]
    for e in elements:
        queries.append(f"What is the chemical symbol for {e}?")
        queries.append(f"What is the atomic number of {e}?")
        queries.append(f"What is the standard state of {e} at room temperature?")

    events = [
        "the French Revolution begin", "the Berlin Wall fall", "the Apollo 11 moon landing happen",
        "World War I end", "World War II begin", "the Magna Carta get signed",
        "the American Declaration of Independence occur", "the Chernobyl disaster occur",
        "the Titanic sink", "the United Nations get founded", "the telescope get invented",
        "the first human travel into space", "the Panama Canal open", "the printing press get invented"
    ]
    for ev in events:
        queries.append(f"In what year did {ev}?")
        queries.append(f"Where did {ev}?")

    books = [
        "1984", "Moby Dick", "To Kill a Mockingbird", "War and Peace", "Pride and Prejudice",
        "The Great Gatsby", "Crime and Punishment", "Don Quixote", "One Hundred Years of Solitude", "Brave New World",
        "The Odyssey", "Hamlet", "Jane Eyre", "Wuthering Heights", "The Catcher in the Rye"
    ]
    for b in books:
        queries.append(f"Who wrote the literary work {b}?")
        queries.append(f"When was {b} first published?")

    scientists = [
        "Albert Einstein", "Marie Curie", "Isaac Newton", "Alan Turing", "Charles Darwin",
        "Ada Lovelace", "Nikola Tesla", "Galileo Galilei", "Alexander Fleming", "Louis Pasteur",
        "Johannes Kepler", "Niels Bohr", "Max Planck", "James Clerk Maxwell", "Rosalind Franklin"
    ]
    for s in scientists:
        queries.append(f"Where was {s} born?")
        queries.append(f"What was {s}'s most significant scientific discovery?")
        queries.append(f"In what field did {s} conduct their primary research?")

    rng.shuffle(queries)
    return list(dict.fromkeys(queries))


def generate_semadpt_samples(seed: int = 42) -> List[str]:
    """
    Generates diverse open-ended creative, stylistic transformation, summarization,
    and open-ended conversational adaptation queries.
    
    Covers:
    A. Creative generation (stories, poems, fictional dialogues, scripts, creative tributes)
    B. Tone transformation (friendly, professional, polite, concise, empathetic rewriting)
    C. Summarization (short summaries, condensing, core takeaways)
    D. Open-ended conversational adaptation (jokes, creative ideation, analogies)
    
    IMPORTANT: None of these queries require strict structural/format schemas or output bounds.
    """
    rng = random.Random(seed)
    queries = []

    # Category A: Creative Generation
    creative_topics = [
        "a rainy evening in a quiet town", "a solitary lighthouse keeper during a storm",
        "an artificial intelligence discovering music", "autumn leaves falling in an ancient courtyard",
        "a bustling night market in Bangkok", "two astronauts lost on a distant moon",
        "the secret life of an old clockmaker", "a stray dog finding a home",
        "an ancient bookstore with infinite shelves", "a sunset viewed from a futuristic skyline",
        "a forgotten violin in an attic", "the first snowfall in a mountain village",
        "friendship between two childhood rivals", "a traveler waiting at an empty train station",
        "the mystery of deep ocean trenches", "a chef preparing a legendary midnight feast"
    ]
    
    # Representative boundary test prompts
    queries.append("Write a short creative story about a rainy evening.")
    queries.append("Create a poem about friendship.")
    queries.append("Generate a fictional dialogue between two students.")

    for t in creative_topics:
        queries.append(f"Write a short story about {t}.")
        queries.append(f"Write a creative story exploring {t}.")
        queries.append(f"Create a poem reflecting on {t}.")
        queries.append(f"Draft an evocative opening paragraph for a novel set around {t}.")
        queries.append(f"Generate a fictional dialogue between two characters discussing {t}.")
        queries.append(f"Write a haiku capturing the atmosphere of {t}.")
        queries.append(f"Compose a descriptive scene illustrating {t}.")
        queries.append(f"Write an imaginative monologue from the perspective of someone witnessing {t}.")

    # Category B: Tone Transformation & Stylistic Rewriting
    scenarios = [
        "apologizing to a client for an unexpected project delay",
        "declining an invitation to speak at a conference due to scheduling conflicts",
        "asking a manager for constructive feedback and a salary review",
        "announcing the successful release of a major product update",
        "giving gentle feedback to a teammate whose presentation lacked focus",
        "explaining to a customer why a warranty claim cannot be processed",
        "welcoming a new colleague to the team during an all-hands meeting",
        "reconnecting with a former colleague after several years",
        "thanking volunteers for their dedication at a community fundraiser",
        "pitching an ambitious research initiative to department leadership",
        "letting a vendor know that their proposal was not selected",
        "reminding employees about the upcoming quarterly review deadline"
    ]
    tones = [
        "friendly", "professional", "polite", "concise",
        "warm and empathetic", "humorous and witty", "persuasive and motivational",
        "diplomatic and constructive", "inspirational", "casual and relaxed"
    ]
    
    # Representative boundary test prompts
    queries.append("Rewrite this sentence in a friendly tone.")
    queries.append("Make this message sound more professional.")
    queries.append("Rewrite this email to sound polite.")
    queries.append("Make this paragraph more concise.")

    for sc in scenarios:
        for tn in tones:
            queries.append(f"Rewrite this message in a {tn} tone: {sc}.")
            queries.append(f"Rewrite this email to sound {tn}: {sc}.")
            queries.append(f"Make this sentence more {tn}: {sc}.")
            queries.append(f"Help me rephrase this message so it feels {tn}: {sc}.")
            queries.append(f"Change the tone of this note to be {tn}: {sc}.")

    # Category C: Summarization
    summary_topics = [
        "the renewable energy transition across global power grids",
        "how mRNA vaccines stimulate the human immune response",
        "the economic principles behind comparative advantage in international trade",
        "recent breakthroughs in quantum computing and error mitigation",
        "the history and architectural legacy of the Silk Road",
        "how sleep deprivation affects memory consolidation in the brain",
        "the ecological significance of mangrove forests in coastal protection",
        "the mechanism of CRISPR-Cas9 targeted genome editing",
        "the lifecycle of stars from nebulae to black holes",
        "the impact of microplastics on marine food webs",
        "the development of urban vertical farming systems",
        "the fundamental concepts behind generative adversarial networks",
        "the physiological benefits of regular aerobic exercise",
        "the geopolitical causes of the First Industrial Revolution",
        "the principles of zero-knowledge cryptographic proofs",
        "the formation and dynamics of oceanic hydrothermal vents",
        "the cognitive psychology behind decision-making heuristics",
        "the history of human space exploration from Apollo to Artemis",
        "the principles of sustainable circular economy design",
        "the cellular mechanisms of aging and telomere attrition"
    ]
    
    # Representative boundary test prompts
    queries.append("Summarize this paragraph.")
    queries.append("Summarize the following paragraph.")
    queries.append("Give me a short summary of this text.")
    queries.append("Condense this passage into two sentences.")

    for st in summary_topics:
        queries.append(f"Summarize the main points of the text regarding {st}.")
        queries.append(f"Give me a short summary of this article on {st}.")
        queries.append(f"Condense the following discussion about {st} into a brief overview.")
        queries.append(f"Provide a concise summary highlighting the key takeaways of {st}.")
        queries.append(f"Summarize the core message of this document about {st}.")
        queries.append(f"Summarize the following passage explaining {st}.")
        queries.append(f"Can you summarize this text about {st} for a general reader?")
        queries.append(f"Condense the technical description of {st} into two sentences.")
        queries.append(f"Explain the concept of {st} using a clear, intuitive analogy.")

    # Category D: Open-Ended Conversational Adaptation & Ideation
    domains = [
        "eco-friendly subscription boxes", "an interactive science podcast for teens",
        "a mobile app connecting urban community gardeners", "remote team-building activities",
        "catchy names for an artisanal sourdough bakery", "compelling themes for an indie fantasy novel",
        "creative ways to practice conversational foreign languages", "memorable taglines for a clean energy startup"
    ]
    
    queries.append("Tell me a joke.")
    queries.append("Can you tell me a funny joke to cheer me up?")
    queries.append("Give me a creative idea.")
    queries.append("Respond casually to this message.")
    queries.append("Hello, how are things going with you today?")

    for d in domains:
        queries.append(f"Brainstorm 5 innovative ideas for {d}.")
        queries.append(f"Suggest creative concepts and fresh angles for {d}.")
        queries.append(f"Give me creative name ideas for {d}.")
        queries.append(f"Help me brainstorm engaging hooks for {d}.")

    rng.shuffle(queries)
    return list(dict.fromkeys(queries))


def generate_cstrset_samples(seed: int = 42) -> List[str]:
    """
    Generates constraint satisfaction, strict schema formatting, item count bounded,
    and rule-bound queries.
    
    PRIMARY INTENT: Satisfying explicit structural/output constraints (JSON, YAML, XML,
    exact item counts, negative constraints, schema rules).
    
    IMPORTANT: Avoids creative writing or stylistic transformation disguised as constraints.
    Focuses entirely on: "What exact structural/output constraints must the response satisfy?"
    """
    rng = random.Random(seed)
    queries = []

    # Representative boundary test prompts
    queries.append("Return the information as JSON with fields name, age, and city.")
    queries.append("Give exactly 3 items in a JSON array and do not include any additional text.")
    queries.append("Give exactly 3 items in a JSON array.")
    queries.append("Return valid YAML only.")
    queries.append("Do not include any additional text.")
    queries.append("Return the answer as JSON with fields name and age.")
    queries.append("Output exactly 3 items.")
    queries.append("Return only the requested values and no explanation.")
    queries.append("The output must follow this schema.")

    # 1. Structured Output Schemas (JSON, YAML, XML, CSV, Markdown Table, SQL)
    schemas = [
        ("user profile", ["user_id", "full_name", "email_address", "created_at", "role"]),
        ("product catalog item", ["sku", "title", "price_usd", "stock_count", "category"]),
        ("weather station report", ["station_id", "temperature_c", "humidity_pct", "wind_kph", "timestamp"]),
        ("flight booking record", ["booking_ref", "passenger_name", "origin_code", "dest_code", "flight_date"]),
        ("movie catalog entry", ["movie_id", "title", "director", "release_year", "rating"]),
        ("student academic transcript", ["student_id", "course_name", "letter_grade", "credit_hours"]),
        ("cloud server telemetry", ["instance_id", "cpu_percent", "memory_used_mb", "disk_free_gb", "status"]),
        ("e-commerce customer order", ["order_id", "customer_id", "total_usd", "payment_status", "item_count"]),
        ("clinical patient record", ["patient_id", "blood_type", "systolic_bp", "heart_rate", "allergies"]),
        ("vehicle fleet telemetry", ["vin", "odometer_km", "fuel_level_pct", "tire_pressure_psi", "speed_kph"]),
        ("commercial invoice", ["invoice_num", "client_name", "issue_date", "subtotal", "tax", "total_due"]),
        ("job application tracking", ["applicant_id", "position_title", "years_experience", "status_code"])
    ]

    for item, keys in schemas:
        k_str = ", ".join(f"'{k}'" for k in keys)
        k_plain = ", ".join(keys)
        
        # JSON formatting constraints
        queries.append(f"Return the {item} data as a valid JSON object with fields: {k_plain}.")
        queries.append(f"Output strictly a JSON object representing a {item} with keys: [{k_str}]. Do not include commentary.")
        queries.append(f"Provide a JSON payload for {item} containing required keys: [{k_str}]. No markdown formatting.")
        queries.append(f"Format the {item} details as valid JSON only matching properties: {k_plain}.")
        queries.append(f"Return a JSON object containing these required fields: [{k_plain}].")

        # YAML formatting constraints
        queries.append(f"Return valid YAML only defining a {item} with keys: [{k_str}].")
        queries.append(f"Produce valid YAML content for a {item} using keys: {k_plain}. Strictly adhere to YAML indentation.")
        queries.append(f"Output as YAML format only: serialize the {item} record with attributes [{k_plain}].")

        # XML formatting constraints
        queries.append(f"Generate a well-formed XML snippet for {item} containing tags: [{k_plain}].")
        queries.append(f"Use XML with these specified tags: {', '.join(f'<{k}>' for k in keys[:3])} to represent a {item}.")
        queries.append(f"Return the {item} record in XML format with root element <{item.replace(' ', '_')}>.")

        # Tabular & Delimited formatting
        queries.append(f"Format data for {item} as a Markdown table with exact column headers: [{k_plain}].")
        queries.append(f"Return only a comma-separated values (CSV) string for {item} with header columns: {k_plain}.")
        queries.append(f"Output a PostgreSQL CREATE TABLE statement for {item} with columns: [{k_plain}].")

    # 2. Explicit Quantity and List Item Count Constraints
    quantities = [2, 3, 4, 5, 6, 7, 8, 10]
    domains = [
        "programming languages", "cloud service providers", "web development frameworks",
        "data visualization libraries", "cybersecurity best practices", "machine learning algorithms",
        "software architectural patterns", "relational database systems", "networking protocols",
        "mobile operating systems", "file compression formats", "version control commands"
    ]
    for q in quantities:
        for d in domains:
            queries.append(f"Output exactly {q} items in a JSON array for {d}.")
            queries.append(f"Give exactly {q} items in a list of {d} and do not include any additional text.")
            queries.append(f"List exactly {q} {d}. No more and no less.")
            queries.append(f"Return a JSON array containing exactly {q} elements of {d}.")
            queries.append(f"Provide exactly {q} bullet points on {d} with no introductory or closing remarks.")

    # 3. Fixed Length & Character Count Constraints (Framed strictly as output constraints)
    topics = [
        "cloud computing architecture", "distributed consensus protocols", "database index structures",
        "asymmetric cryptography", "container orchestration systems", "microservice design patterns",
        "HTTP/2 protocol multiplexing", "garbage collection algorithms", "load balancing algorithms",
        "domain name system resolution", "content delivery networks", "memory hierarchy in modern CPUs"
    ]
    for t in topics:
        queries.append(f"Provide an explanation of {t} in exactly 50 words.")
        queries.append(f"Explain {t} using fewer than 25 words total.")
        queries.append(f"Describe {t} in between 30 and 40 words total.")
        queries.append(f"Limit the output for {t} strictly to fewer than 30 words.")
        queries.append(f"Limit the explanation of {t} strictly to under 140 characters, including spaces.")
        queries.append(f"Provide a 1-sentence technical definition of {t} containing at least 20 words.")

    # 4. Strict Negative Constraints & Exclusion Rules
    prohibited_sets = [
        ("a relational database", ["table", "sql", "query", "row"]),
        ("a web browser", ["internet", "page", "tab", "url"]),
        ("cloud storage", ["file", "server", "data", "upload"]),
        ("a computer network", ["cable", "router", "ip", "packet"]),
        ("a microservice", ["monolith", "api", "docker", "deploy"]),
        ("a compiler", ["code", "binary", "syntax", "language"]),
        ("an operating system", ["kernel", "process", "memory", "driver"]),
        ("a cryptographic hash", ["sha", "md5", "digest", "encryption"]),
        ("a message queue", ["kafka", "rabbit", "topic", "consumer"]),
        ("a cache system", ["redis", "memcache", "hit", "ttl"])
    ]
    for target, words in prohibited_sets:
        w_str = ", ".join(f"'{w}'" for w in words)
        queries.append(f"Define {target} while completely avoiding the following forbidden words: {w_str}.")
        queries.append(f"Describe {target} under the negative constraint: you are strictly forbidden from using {w_str}.")
        queries.append(f"Explain {target} without using any of the terms: {w_str}. Output only the definition.")
        queries.append(f"State the function of {target} while omitting all words from this list: {w_str}.")

    # 5. Rule-Bound Syntax, Pattern Enforcement & Delimiter Rules
    patterns = [
        "Ensure every line starts with a timestamp in format [YYYY-MM-DD HH:MM:SS]",
        "Output each item as 'ID: <UUID>, Value: <Integer>, Status: <ACTIVE|INACTIVE>'",
        "Generate output strictly adhering to regex pattern: ^[A-Z]{3}-[0-9]{4}-[A-Z]$",
        "Format the response such that every line contains exactly one key-value pair separated by an equals sign (=)",
        "Prefix every line with a sequential numbering like [1], [2], [3]",
        "Format each line as JSONL: exactly one valid JSON object per line with keys 'step' and 'action'",
        "Output using only capitalized words (ALL CAPS) with no lowercase letters",
        "Every line must strictly terminate with a semicolon (;)",
        "Wrap every value in double quotes and separate entries with a pipe delimiter (|)",
        "The output must follow this schema: key: string, count: integer, active: boolean"
    ]
    for p in patterns:
        for t in topics[:8]:
            queries.append(f"Generate data for {t} strictly adhering to rule: {p}.")
            queries.append(f"Output info on {t} under this strict formatting constraint: {p}.")
            queries.append(f"Produce output for {t} enforcing: {p}. No commentary or extra text.")

    rng.shuffle(queries)
    return list(dict.fromkeys(queries))
