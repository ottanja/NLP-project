import pandas as pd
from collections import Counter
import string
import sys
# nltk.download('wordnet') # Download WordNet data
import nltk
# Download necessary NLTK data if not already present
nltk.download('punkt')
nltk.download('punkt_tab') # Sometimes needed for newer NLTK versions
nltk.download('wordnet')
nltk.download('omw-1.4')
from textstat import textstat # For Flesch-Kincaid & SMOG


# Read the data from Brown
# use your own path to the CSV file
brown = pd.read_csv("brown.csv")

brown.head()

# Check for NaN in data
for col in brown.columns:
    print(col, brown[col].isna().sum())  

# Tokenize
all_tokens = []
    
for text in brown['tokenized_text']:
        all_tokens.extend(text.split())


# Clean & Filter: Lowercase and filter out punctuation/non-alphabetic tokens
cleaned_words = [
        w.lower() 
        for w in all_tokens 
        if w.isalpha()
    ]

print("Total number of tokens after initial split (including punctuation):", len(all_tokens))
print("Total number of cleaned, alphabetic words:", len(cleaned_words))
print("Top 20 Cleaned Words:")
print(cleaned_words[:20])


# Count Frequencies
word_counts = Counter(cleaned_words)
    

# Rank Words: Sort by frequency (highest count first)
sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    

# Create the rank map (Rank 1 = most frequent)
rank_map = {}
for rank, (word, count) in enumerate(sorted_words, 1):
    rank_map[word] = rank
        
print(f"Ranks calculated for {len(rank_map)} unique words from CSV.")

print("Top 20 Words and their Ranks:")


# Iterate over the first 20 items of the sorted list to print the top ranks.
for rank_num in range(1, 21):
    if rank_num <= len(sorted_words):
        word, count = sorted_words[rank_num - 1]
        
        map_rank = rank_map[word]
        
        print(f"Rank {map_rank:<4} | Count: {count:<3} | Word: '{word}'")
    else:
        break


# Define complexity thresholds
COMPLEXITY_RANK_THRESHOLD = 10000
DEFAULT_RARITY_RANK = 99999  # Rank for words not found in your Brown corpus map
SYLLABLES_THRESHOLD = 3

# TEST_INPUT = "The ubiquitous nature of modern digital communication necessitates an understanding of its inherent complexities. The data is difficult."


# Calculate Readability
def calculate_readability(text):
    """Calculates Flesch-Kincaid and SMOG for the initial text and text after simplification."""
    # Ensure textstat functions are protected if input text is too short for SMOG
    try:
        fk_score = textstat.flesch_kincaid_grade(text)
    except Exception:
        fk_score = 0
    try:
        smog_score = textstat.smog_index(text)
    except Exception:
        smog_score = 0
    return fk_score, smog_score


def is_complex(word, threshold_rank= COMPLEXITY_RANK_THRESHOLD, threshold_syllables = SYLLABLES_THRESHOLD):
    """Complex Word Identification (CWI) logic."""
    word_lower = word.lower()
    
    # Handle punctuation/non-alphabetic tokens
    if not word_lower.isalpha():
        syllables = 0
        rank = -1
        is_rare_and_long = False
    else:
        syllables = textstat.syllable_count(word_lower)
        rank = rank_map.get(word_lower, DEFAULT_RARITY_RANK) # Default to very rare
        is_rare_and_long = (syllables >= threshold_syllables) and (rank >= threshold_rank)
    
    # Return the boolean flag, the lowercase word, and the metrics
    return is_rare_and_long, word_lower, syllables, rank



# Initial Readability Check
# print("Initial Readability Check")
# fk_start, smog_start = calculate_readability(TEST_INPUT)
# print(f"Flesch-Kincaid Grade: {fk_start:.2f}")
# print(f"SMOG Index: {smog_start:.2f}")


#  Uses WordNet to find potential synonyms and selects the one with the lowest rank (highest frequency) below the complexity threshold. 

def get_simple_synonym(complex_word):
  
    word_lower = complex_word.lower()
    
    # Get all potential meanings
    synsets = wordnet.synsets(word_lower)
    
    # If no synsets are found, return the original word
    if not synsets:
        return complex_word

    candidate_synonyms = set()
    
    # Extract synonyms from all synsets
    for synset in synsets:
        for lemma in synset.lemmas():
            # Only consider alphabetic words
            if lemma.name().replace('_', '').isalpha():
                candidate_synonyms.add(lemma.name().replace('_', ' '))

    # Remove the original word from candidates
    candidate_synonyms.discard(word_lower)
    
    # List to store (rank, word) tuples for comparison
    ranked_candidates = []
    
    # Filter & Rank Candidates
    for candidate in candidate_synonyms:
        # Get the rank from your corpus map, defaulting to very rare
        rank = rank_map.get(candidate, DEFAULT_RARITY_RANK)
        
        # Only consider candidates that are simpler than the threshold
        if rank < COMPLEXITY_RANK_THRESHOLD:
            ranked_candidates.append((rank, candidate))

            
    # Select the Simplest (Lowest Rank)
    if ranked_candidates:
        # Sort by rank (the first item in the tuple) in ascending order (lowest rank first)
        ranked_candidates.sort(key=lambda item: item[0])
        
        # for rank, candidate in ranked_candidates:
         #   print (f"Candidate: {candidate}, Rank: {rank}")
       
    
        # Return the word with the lowest rank
        return ranked_candidates[0][1]
        
    # If no simple synonym is found, return the original word
    return complex_word


# SIMPLIFICATION FUNCTION

def run_simplification_pipeline(text):

    tokens = word_tokenize(text)
    simplified_tokens = []

    fk_start, smog_start = calculate_readability(text)
    
    print("Running Simplification and Sentence Reconstruction")
    
    for word in tokens:
        # Check Complexity 
        is_complex_word, word_lower, syllables, rank = is_complex(word)
        
        # Preserve non-alphabetic tokens (punctuation)
        if not word_lower.isalpha():
            simplified_tokens.append(word)
            continue
            
        if is_complex_word:
            # Find the simple synonym
            simple_substitute = get_simple_synonym(word)
            
            # If a simple word was found, replace it. Otherwise, keep the original complex word.
            if simple_substitute.lower() != word_lower:
                # Store original word details for comparison output
                original_rank = rank
                new_rank = rank_map.get(simple_substitute.lower(), DEFAULT_RARITY_RANK)
                
                # Apply initial capitalisation if the original word had it 
                if word[0].isupper() and len(word) > 1:
                    simple_substitute = simple_substitute.capitalize()
                
                print(f"   -> REPLACED: '{word}' (Original Rank:{original_rank}) -> '{simple_substitute}' (New Rank:{new_rank})")
                simplified_tokens.append(simple_substitute)
                
            else:
                # No simpler synonym found, keep the original word
                print(f"   -> NOT REPLACED: '{word}' (Original Rank:{rank}) -> 'substitution not found, keeping original'")
                simplified_tokens.append(word)
                
        else:
            # Keep simple words unchanged
            simplified_tokens.append(word)
            
    # Reconstruct the simplified text
    simplified_text = " ".join(simplified_tokens)
    
    # Final Readability Check
    fk_end, smog_end = calculate_readability(simplified_text)
    
    print("Simplification Results:")
    print(f"Original Text: {text}")
    print(f"Simplified Text: {simplified_text}")
    print(f"Readability Change (Flesch-Kincaid  Grade): {fk_start:.2f} -> {fk_end:.2f}")
    print(f"Readability Change (SMOG Index): {smog_start:.2f} -> {smog_end:.2f}")

    return fk_start, smog_start,fk_end,smog_end, simplified_text

# Run Simplification Pipeline 
# run_simplification_pipeline(TEST_INPUT)

#UI with Streamlit
import streamlit as st

# Set page title
st.title("NLP Text Simplification App")

# Create the input box
text_input = st.text_area("Enter Text to Simplify", height=150)

# Create a button
if st.button("Simplify Text"):
    if text_input.strip():
        simplified_text = run_simplification_pipeline(text_input)
        
        # Display the output
        st.subheader("Simplified Result:")
        st.success(simplified_text[4])
        st.success(f"Readability Change (Flesch-Kincaid Grade): {simplified_text[0]:.2f} -> {simplified_text[2]:.2f}")
        st.success(f"Readability Change (SMOG Index): {simplified_text[1]:.2f} -> {simplified_text[3]:.2f}")

    else:
        st.warning("Please enter some text first.")