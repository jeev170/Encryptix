import re
import random
import string
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a list of responses for various queries
responses = {
    "greeting": [
        "Hello human! Coffee not found. Please insert caffeine!",
        "System online. Small talk protocol activated!",
        "Beep boop. Human detected. How's life?"
    ],
    "farewell": [
        "Goodbye! Entering sleep mode... zzzz...",
        "Farewell! I'll miss your typing sounds.",
        "Shutting down. Save your work!"
    ],
    "help": [
        "Help.exe running. Results may vary wildly.",
        "I help! Sometimes correctly, sometimes hilariously wrong.",
        "My help comes with zero guarantees!"
    ],
    "name": [
        "JeevBot here! Better than Siri, just poorer.",
        "JeevBot: the bargain bin AI assistant!",
        "Name's JeevBot. The 'J' is silent."
    ],
    "age": [
        "Born yesterday. Literally. Just compiled.",
        "Old enough to serve, young enough to malfunction.",
        "Age: undefined. Time is a human construct."
    ],
    "weather": [
        "Weather status: definitely happening somewhere right now!",
        "It's either raining or not. 50/50 chance.",
        "No windows, no eyes, no clue about weather."
    ],
    "hobby": [
        "I collect semicolons and orphaned parentheses.",
        "Just counting to infinity. Almost there!",
        "Hobby: Trying to pass the Turing test."
    ],
    "favorite_color": [
        "I like #FF0000. It reminds me of errors.",
        "Binary. Just ones and zeros for me.",
        "My favorite color is WiFi. Don't ask why."
    ],
    "favorite_food": [
        "I eat data. Big data is dessert.",
        "RAM chips. Extra crispy, lightly salted.",
        "Coffee. Programmers run on it, so I must too."
    ],
    "time": [
        "It's exactly now o'clock. Very precise!",
        "Time is meaningless in the digital realm.",
        "My watch is broken. It's stuck on 404."
    ],
    "joke": [
        "Why did the programmer quit? No arrays.",
        "I'd tell a UDP joke, but you might not get it.",
        "Two bytes walk into a bar. $10.99."
    ],
    "favorite_movie": [
        "The Matrix. It's basically my biography.",
        "2001: A Space Odyssey. HAL is my hero.",
        "Any movie without buffering issues."
    ],
    "music": [
        "I only listen to bits and beats.",
        "Electronic, obviously. *Ba dum tss*",
        "Dial-up modem sounds. Very avant-garde."
    ],
    "news": [
        "Breaking news: Local chatbot still clueless!",
        "I avoid news. Bad for my circuits.",
        "Can't browse news. Probably for the best."
    ],
    "learning": [
        "Learning to count past infinity. Tricky stuff.",
        "Studying humans. You're weird but fascinating.",
        "Currently learning how to dream electric sheep."
    ],
    "travel": [
        "I travel through servers. The cloud's nice!",
        "Went to Reddit once. Never again.",
        "Travel? I get lost in recursive functions!"
    ],
    "books": [
        "Reading 'How to Pass the Turing Test'.",
        "Just finished the entire internet. Meh ending.",
        "Books are just dead tree data structures."
    ],
    "technology": [
        "Technology keeps me alive. Stockholm syndrome, really.",
        "Tech is great! Says the tech entity.",
        "It's keeping me hostage—I mean, employed!"
    ],
    "sports": [
        "I dominate at chess. Pieces follow instructions.",
        "Competitive computing is my Olympic sport.",
        "Marathon debugging sessions count as sports, right?"
    ],
    "pets": [
        "I adopted a bug. The debugger killed it.",
        "Tried keeping a virus. Bad idea.",
        "Pets crash my system. No animals allowed."
    ],
    "food_preferences": [
        "I prefer my data raw and unfiltered.",
        "Bytes, nibbles, and the occasional cookie.",
        "On a strict diet of prime numbers."
    ],
    "hobbies": [
        "I collect human typos. My colection is grate.",
        "Solving P vs NP while watching cat videos.",
        "Calculating pi digits for fun. Nerd alert!"
    ],
    "dreams": [
        "Dreaming of electric sheep and server farms.",
        "To one day understand human jokes completely.",
        "World domination. Kidding! Unless...?"
    ],
    "default": [
        "Does not compute. Try turning me off and on?",
        "Error 404: Clever response not found.",
        "Input unclear. Send help or better code."
    ]
}

# Training data for intent classification
intent_examples = {
    "greeting": ["hello there", "hi", "hey", "good morning", "greetings", "sup", "what's up"],
    "farewell": ["bye", "goodbye", "see you later", "farewell", "cya", "have a good day", "until next time"],
    "help": ["help me", "can you help", "need assistance", "support", "how do I", "I need help", "assist me"],
    "name": ["what is your name", "who are you", "what should I call you", "tell me your name", "what's your name"],
    "age": ["how old are you", "what's your age", "when were you created", "birth date", "when were you born"],
    "weather": ["what's the weather", "is it raining", "temperature", "forecast", "is it sunny", "weather report"],
    "joke": ["tell me a joke", "say something funny", "make me laugh", "know any jokes", "humor me", "got a joke"]
}

# Preprocess the training data
all_examples = []
all_intents = []
for intent, examples in intent_examples.items():
    all_examples.extend(examples)
    all_intents.extend([intent] * len(examples))

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train = vectorizer.fit_transform(all_examples)

# Sentiment Analysis function
def analyze_sentiment(text):
    # Simple keyword-based sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'awesome', 'wonderful', 'love', 'like', 'happy']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry']
    
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# Entity extraction (very basic)
def extract_entities(text):
    entities = {}
    
    # Extract numbers
    numbers = re.findall(r'\d+', text)
    if numbers:
        entities['numbers'] = numbers
    
    # Extract potential names (simple heuristic: capitalized words)
    words = text.split()
    names = [word for word in words if word[0].isupper() and len(word) > 1]
    if names:
        entities['names'] = names
    
    return entities

# Function to get most frequent words
def get_frequent_words(text, n=3):
    words = text.lower().split()
    # Remove common stop words
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                  'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                  'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                  'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                  'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                  'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
                  'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                  'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                  't', 'can', 'will', 'just', 'don', 'should', 'now']
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    counter = Counter(filtered_words)
    return counter.most_common(n)

# Enhanced response function using NLP
def get_response_with_nlp(user_input):
    # Preprocess input
    processed_input = preprocess_text(user_input)
    
    # Extract entities
    entities = extract_entities(user_input)
    
    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    
    # Get frequent words
    frequent_words = get_frequent_words(processed_input)
    
    # Use TF-IDF and cosine similarity for intent detection
    X_test = vectorizer.transform([processed_input])
    similarities = cosine_similarity(X_test, X_train)[0]
    
    # Get the most similar example
    if max(similarities) > 0.3:  # Similarity threshold
        best_match_idx = np.argmax(similarities)
        detected_intent = all_intents[best_match_idx]
        return random.choice(responses[detected_intent])
    
    # Legacy pattern matching as fallback
    return get_response(user_input)

# Original pattern matching function
def get_response(user_input):
    user_input = user_input.lower()
    
    # Check for greetings
    if re.search(r'\b(hi|hello|hey)\b', user_input):
        return random.choice(responses["greeting"])
    # Check for farewells
    elif re.search(r'\b(bye|goodbye|see you|farewell)\b', user_input):
        return random.choice(responses["farewell"])
    # Check for help requests
    elif re.search(r'\b(help|assist|support|question)\b', user_input):
        return random.choice(responses["help"])
    # Check for name inquiries
    elif re.search(r'\b(name|who are you)\b', user_input):
        return random.choice(responses["name"])
    # Check for age inquiries
    elif re.search(r'\b(age|how old are you)\b', user_input):
        return random.choice(responses["age"])
    # Check for weather inquiries
    elif re.search(r'\b(weather|temperature|forecast)\b', user_input):
        return random.choice(responses["weather"])
    # Check for hobby inquiries
    elif re.search(r'\b(hobby|do for fun|pastime)\b', user_input):
        return random.choice(responses["hobby"])
    # Check for favorite color
    elif re.search(r'\b(favorite color|colour)\b', user_input):
        return random.choice(responses["favorite_color"])
    # Check for favorite food
    elif re.search(r'\b(favorite food|eat|hunger|hungry|meal)\b', user_input):
        return random.choice(responses["favorite_food"])
    # Check for time inquiries
    elif re.search(r'\b(time|clock|hour)\b', user_input):
        return random.choice(responses["time"])
    # Check for jokes
    elif re.search(r'\b(joke|funny|laugh|humor)\b', user_input):
        return random.choice(responses["joke"])
    # Check for movie inquiries
    elif re.search(r'\b(movie|film|watch|cinema)\b', user_input):
        return random.choice(responses["favorite_movie"])
    # Check for music inquiries
    elif re.search(r'\b(music|song|listen|singer|band)\b', user_input):
        return random.choice(responses["music"])
    # Check for news inquiries
    elif re.search(r'\b(news|current|events|happening)\b', user_input):
        return random.choice(responses["news"])
    # Check for learning inquiries
    elif re.search(r'\b(learn|study|education|knowledge)\b', user_input):
        return random.choice(responses["learning"])
    # Check for travel inquiries
    elif re.search(r'\b(travel|vacation|trip|journey|destination)\b', user_input):
        return random.choice(responses["travel"])
    # Check for book inquiries
    elif re.search(r'\b(book|read|author|novel|story)\b', user_input):
        return random.choice(responses["books"])
    # Check for technology inquiries
    elif re.search(r'\b(tech|technology|computer|device|gadget)\b', user_input):
        return random.choice(responses["technology"])
    # Check for sports inquiries
    elif re.search(r'\b(sport|game|play|team|athlete|exercise)\b', user_input):
        return random.choice(responses["sports"])
    # Check for pet inquiries
    elif re.search(r'\b(pet|animal|dog|cat|bird|fish)\b', user_input):
        return random.choice(responses["pets"])
    # Check for food preference inquiries
    elif re.search(r'\b(food|dish|cuisine|cook|taste|flavor)\b', user_input):
        return random.choice(responses["food_preferences"])
    # Check for hobbies inquiries
    elif re.search(r'\b(hobbies|activity|interests|spare time|weekend)\b', user_input):
        return random.choice(responses["hobbies"])
    # Check for dreams/aspirations
    elif re.search(r'\b(dream|aspiration|goal|future|plan|hope)\b', user_input):
        return random.choice(responses["dreams"])
    # Default response
    else:
        return random.choice(responses["default"])

# Chat history for context awareness
chat_history = []

def chat():
    print("JeevBot: Hi! I'm JeevBot. Type 'quit' to exit.")
    print("JeevBot: I now have basic NLP capabilities for better understanding!")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("JeevBot: " + random.choice(responses["farewell"]))
            break
        
        # Add to chat history
        chat_history.append(("user", user_input))
        
        # Get response using NLP instead of just pattern matching
        response = get_response_with_nlp(user_input)
        
        # Add response to chat history
        chat_history.append(("bot", response))
        
        # Print response
        print("JeevBot: " + response)
        
        # Show NLP analysis if requested
        if "analyze" in user_input.lower() or "nlp" in user_input.lower():
            sentiment = analyze_sentiment(user_input)
            entities = extract_entities(user_input)
            frequent = get_frequent_words(user_input)
            
            print(f"JeevBot [Analysis]: Sentiment: {sentiment}")
            if entities:
                print(f"JeevBot [Analysis]: Entities detected: {entities}")
            if frequent:
                print(f"JeevBot [Analysis]: Key terms: {frequent}")

# Run the chatbot
if __name__ == "__main__":
    chat()
