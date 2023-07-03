from transformers import pipeline
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, RegexpParser

# Set the path to the keywords file
keywords_file = 'keywords.txt'

# Set the path to the INPUT folder
input_folder = 'INPUT'

# Set the path to the CODING folder
coding_folder = 'CODING'

# Set the path to the pairs.txt file
pairs_file = 'pairs.txt'

# Function to check for the first non-empty file in the INPUT folder

def get_first_non_empty_file(input_folder):
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            return file_path
    return None

# Function to break text into sentences

def break_text_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Function to determine the type of sentence (question, exclamation, declaration, etc.)

def determine_sentence_type(sentence):
    if sentence.endswith('?'):
        return 'question'
    elif sentence.endswith('!'):
        return 'exclamation'
    else:
        return 'declaration'

# Function to get keywords from a sentence

def get_keywords(sentence):
    # Define the stop words
    stop_words = set(stopwords.words('english'))

    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Extract the keywords
    keywords = []
    for word in words:
        # Remove punctuation and convert to lowercase
        word = word.strip('.,!?()[]{}:;\'"').lower()

        # Skip stop words and short words
        if word in stop_words or len(word) < 3:
            continue

        # Lemmatize the word
        word = lemmatizer.lemmatize(word)

        # Add the word to the list of keywords
        keywords.append(word)

    # Append the keywords to the keywords.txt file
    with open('keywords.txt', 'a') as f:
        f.write('\n'.join(keywords))

    return keywords


# Function to generate grammatically correct sentences as fallback reply

def generate_fallback_reply(keywords):
    # Initialize the text generation pipeline
    text_generator = pipeline('text-generation', model='gpt2')

    # Generate text using the provided keywords as a prompt
    text = text_generator(' '.join(keywords))[0]['generated_text']

    # Set the fallback reply variable to the generated text
    fallback_reply = text

    return fallback_reply


# Function to check if a sentence is a coding question

def is_coding_question(sentence, keywords):
    # Set the path to the file that contains the bias flag
    bias_file = 'bias.txt'

    # Read the bias flag from the file
    with open(bias_file, 'r') as f:
        bias_flag = f.read(1)

    # Set the bias towards coding or non-coding questions based on the bias flag
    if bias_flag == '1':
        coding_bias = 0.3
    else:
        coding_bias = 0.7

    # Define a list of coding-related keywords
    coding_keywords = ['code', 'coding', 'program',
                       'programming', 'algorithm', 'script', 'function', 'method']

    # Count the number of coding-related keywords in the sentence
    coding_keyword_count = 0
    for keyword in keywords:
        if keyword.lower() in coding_keywords:
            coding_keyword_count += 1

    # Calculate the ratio of coding-related keywords to total keywords
    coding_keyword_ratio = coding_keyword_count / len(keywords)

    # Determine if the sentence is a coding question based on the calculated ratio and the coding bias
    if coding_keyword_ratio >= coding_bias:
        return True
    else:
        return False


# Function to save keywords to the first empty file in the CODING folder


def save_keywords_to_coding_folder(keywords, coding_folder):
    for file in os.listdir(coding_folder):
        file_path = os.path.join(coding_folder, file)
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            with open(file_path, 'w') as f:
                f.write('\n'.join(keywords))
            break

# Function to compare phrases with lines in pairs.txt file and generate reply sentences


def generate_reply_sentences(phrases, pairs_file):
    # Define the maximum number of matches to consider
    max_matches = 5

    # Read the lines from the pairs.txt file
    with open(pairs_file, 'r') as f:
        pairs_lines = f.readlines()

    # Initialize a list to store the matches
    matches = []

    # Iterate over the phrases
    for phrase in phrases:
        # Initialize a list to store the scores for this phrase
        scores = []

        # Iterate over the lines in the pairs.txt file
        for line in pairs_lines:
            # Split the line into two parts using the "abcdefghij" separator
            parts = line.split('abcdefghij')

            # Check if the line has two parts
            if len(parts) == 2:
                # Calculate the score for this line by counting the number of times the phrase appears in the first part of the line
                score = parts[0].count(phrase)

                # Add the score and the second part of the line to the list of scores
                scores.append((score, parts[1]))

        # Sort the scores in descending order by score
        scores.sort(key=lambda x: x[0], reverse=True)

        # Add the top max_matches matches to the list of matches
        matches.extend([score[1] for score in scores[:max_matches]])

    # Generate reply sentences from matches
    reply_sentences = []
    for match in matches:
        reply_sentence = match  # Placeholder value
        reply_sentences.append(reply_sentence)

    return reply_sentences

# Function to make reply sentences more humane

def make_reply_sentences_more_humane(reply_sentences):
    # Initialize the text generation pipeline
    text_generator = pipeline('text-generation', model='gpt2')

    # Join the reply sentences into a single text
    text = ' '.join(reply_sentences)

    # Generate additional text using the reply sentences as a prompt
    additional_text = text_generator(text)[0]['generated_text']

    # Split the additional text into sentences
    additional_sentences = additional_text.split('. ')

    # Combine the original reply sentences with the additional sentences
    combined_sentences = reply_sentences + additional_sentences

    return combined_sentences

# Function to check and correct individual reply sentences


def check_and_correct_sentences(sentences, pairs_file, keywords_file):
    # Define the minimum and maximum sentence length
    min_sentence_length = 3
    max_sentence_length = 20

    # Read the lines from the pairs.txt file
    with open(pairs_file, 'r') as f:
        pairs_lines = f.readlines()

    # Read the lines from the keywords.txt file
    with open(keywords_file, 'r') as f:
        keywords_lines = f.readlines()

    # Initialize a list to store the corrected sentences
    corrected_sentences = []

    # Iterate over the sentences
    for sentence in sentences:
        # Split the sentence into words
        words = sentence.split()

        # Check if the sentence is too short or too long
        if len(words) < min_sentence_length or len(words) > max_sentence_length:
            continue

        # Check if the sentence is a duplicate
        if sentence in corrected_sentences:
            continue

        # Check if the sentence is too similar to a sentence in the pairs.txt file
        too_similar = False
        for line in pairs_lines:
            if line.strip() in sentence:
                too_similar = True
                break
        if too_similar:
            continue

        # Check if the sentence contains too similar keywords
        too_similar_keywords = False
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                too_similar_keywords = True
                break
        if too_similar_keywords:
            continue

        # Replace keywords that are not present in the keywords.txt file
        for i in range(len(words)):
            keyword_found = False
            for line in keywords_lines:
                if words[i] in line.split(','):
                    keyword_found = True
                    break
            if not keyword_found:
                words[i] = '<REPLACEMENT>'

        # Reconstruct the sentence from the corrected words
        corrected_sentence = ' '.join(words)

        # Add punctuation between sentences (alternating between commas, full stops, and semicolons)
        if len(corrected_sentences) % 3 == 0:
            corrected_sentence += '.'
        elif len(corrected_sentences) % 3 == 1:
            corrected_sentence += ','
        else:
            corrected_sentence += ';'

        # Add the corrected sentence to the list of corrected sentences
        corrected_sentences.append(corrected_sentence)

    return corrected_sentences

def extract_phrases(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Perform POS tagging on the words
    tagged_words = pos_tag(words)

    # Define a regular expression pattern for phrase extraction
    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+}   # Noun phrase
        """

    # Create a chunk parser with the defined grammar pattern
    chunk_parser = RegexpParser(grammar)

    # Parse the tagged words to extract noun and adjective phrases
    tree = chunk_parser.parse(tagged_words)

    # Initialize a list to store the extracted phrases
    phrases = []

    # Traverse the tree to extract the phrases
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            # Extract the words in the phrase
            phrase_words = [lemmatizer.lemmatize(word.lower()) for word, tag in subtree.leaves()]

            # Remove stop words and short words from the phrase
            phrase_words = [word for word in phrase_words if word not in stop_words and len(word) >= 3]

            # Combine the words to form the phrase
            phrase = ' '.join(phrase_words)

            # Add the phrase to the list of extracted phrases
            phrases.append(phrase)

    return phrases


while True:
    # Get the first non-empty file in the INPUT folder
    input_file = get_first_non_empty_file(input_folder)

    if input_file:
        # Read the content of the input file
        with open(input_file, 'r') as f:
            text = f.read()

        # Break the text into sentences
        sentences = break_text_into_sentences(text)

        for sentence in sentences:
            # Determine the type of sentence (question, exclamation, declaration, etc.)
            sentence_type = determine_sentence_type(sentence)

            # Get keywords from the sentence
            keywords = get_keywords(sentence)

            # Generate fallback reply using keywords
            fallback_reply = generate_fallback_reply(keywords)

            # Check if the sentence is a coding question
            if is_coding_question(sentence, keywords):
                # Save keywords to the first empty file in the CODING folder and stop processing this sentence
                save_keywords_to_coding_folder(keywords, coding_folder)
                continue

            # Break sentence into phrases and get rid of non-keyword parts (conjunctions, prepositions, etc.)
            phrases = extract_phrases(sentence)

            # Compare phrases with lines in pairs.txt file and generate reply sentences
            reply_sentences = generate_reply_sentences(phrases, pairs_file)

            # Make reply sentences more humane by adding sentences before, in between and after original sentences
            reply_sentences = make_reply_sentences_more_humane(reply_sentences)

            # Check and correct individual reply sentences by adding conjunctions, mixing short and long sentences, adding paragraphs etc.
            reply_sentences = check_and_correct_sentences(reply_sentences, pairs_file, keywords_file)

            # Append resultant string to original input file
            with open(input_file, 'a') as f:
                f.write('\n'.join(reply_sentences))

