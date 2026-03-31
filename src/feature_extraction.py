import pandas as pd
import spacy
import re
import sys
from pathlib import Path
from textblob import TextBlob          
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DISAGREEMENT_FILE, PROC_DIR, FEATURES_FILE

# Download vader lexicon if not already downloaded
nltk.download('vader_lexicon', quiet=True)

df = pd.read_csv(DISAGREEMENT_FILE)

# FIX: keep parser enabled (needed for dep_depth), only disable ner + lemmatizer
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

# Initialise VADER once outside the function — expensive to create repeatedly
vader = SentimentIntensityAnalyzer()

negation_words = {
    # Core negations
    "not", "no", "never", "neither", "nor",
    "nobody", "nothing", "nowhere", "none",
    # Contraction suffix (covers don't, won't, can't, isn't, wasn't etc.)
    "n't",
    # Near-negations (barely positive = functionally negative)
    "hardly", "barely", "scarcely", "rarely", "seldom",
    # Implicit negation verbs
    "lack", "lacks", "lacking", "fail", "fails", "failed",
    "refuse", "refuses", "refused", "deny", "denies", "denied",
    "without", "absence",
}

hedge_words = {
    # Core hedges
    "maybe", "perhaps", "might", "could", "possibly",
    "probably", "seems", "appears", "may", "should", "would",
    # Epistemic verbs — specific enough to be genuine hedges
    "believe", "suppose", "guess", "assume",
    "suspect", "reckon",
    # Approximators — specific enough
    "somewhat", "fairly", "rather", "quite", "pretty",
    "roughly", "approximately",
    # Attribution shields
    "apparently", "allegedly", "supposedly", "seemingly",
    "reportedly", "presumably",
    # Vague frequency — kept only the less common ones
    "sometimes", "occasionally",
}

# 1. Explicit uncertainty markers — person is stating they don't know
uncertainty_words = {
    "unclear", "unsure", "uncertain", "confused", "confusing",
    "ambiguous", "vague", "mixed", "complicated", "complex",
    "depends", "depending", "whatever", "whichever",
    "somehow", "something", "someone", "somewhere",
    "unknown", "unresolved", "undecided", "unexplained",
    "puzzling", "puzzled", "baffling", "baffled",
    "perplexed", "bewildered", "conflicted", "torn",
    "weird", "strange", "odd", "peculiar",
    "debatable", "questionable", "doubtful", "dubious",
    "controversial", "contested",
}

# 2. Contrast markers — mixed signals in one sentence
# "I love it BUT it's terrible" — this is where annotators split
contrast_words = {
    "but", "however", "yet", "although", "though",
    "despite", "while", "whereas", "nevertheless",
    "nonetheless", "still", "except", "unless", "rather",
    "admittedly", "granted", "instead", "alternatively",
    "conversely", "surprisingly", "unexpectedly",
    "ironically", "paradoxically", "oddly", "strangely",
    "partially", "partly", "mostly", "anymore",
    "formerly", "previously", "originally",
}

intensity_words = {
    "absolutely", "literally", "totally", "completely", "utterly",
    "extremely", "incredibly", "insanely", "ridiculously", "so",
    "very", "really", "truly", "deeply", "seriously", "honestly",
    "genuinely", "actually", "just",
}

sarcasm_pattern = re.compile(
    # --- Punctuation signals ---
    r'[*"\']{1,2}\w+[*"\']{1,2}'   # *word*, "word", 'word'
    r'|\.{3,}'                       # ellipsis ... or ....
    r'|!{2,}'                        # !!! multiple exclamation
    r'|\?{2,}'                       # ??? repeated questioning
    r'|\?!'                          # ?! disbelief
    r'|!\?'                          # !? disbelief

    # --- Typographic emphasis ---
    r'|_{1,2}\w+_{1,2}'             # _word_ or __word__
    r'|\b[A-Z]{3,}\b'               # SHOUTING (3+ consecutive caps)
    r'|(?:[A-Z][a-z]){2,}'          # SaRcAsM alternating caps

    # --- Explicit sarcasm tags (Reddit specific) ---
    r'|/s\b'                         # /s
    r'|/sarc\b',                     # /sarc

    re.IGNORECASE
)

def extract_features_from_doc(doc, original_text):
    # collect lowercase tokens and POS tags together
    lower_tokens = [t.text.lower() for t in doc if not t.is_space]
    pos_tags     = [t.pos_ for t in doc]
    n            = len(lower_tokens) or 1  # FIX: avoid division by zero below
    
    # VADER scores — designed for social media text
    # neg, neu, pos → proportion of text that is negative/neutral/positive
    # compound      → overall sentiment strength -1.0 to +1.0
    vs = vader.polarity_scores(original_text)

    return {
        # --- Surface features ---
           "sent_length":       len(doc),
           "negation_count":    sum(1 for t in lower_tokens if t in negation_words),
           "hedge_count":       sum(1 for t in lower_tokens if t in hedge_words),
           "pronoun_count":     pos_tags.count("PRON"),
           "verb_count":        pos_tags.count("VERB"),
       
        # --- Instead of ambiguity_count - two precise features ---
            "uncertainty_count": sum(1 for t in lower_tokens if t in uncertainty_words),
            "contrast_count":    sum(1 for t in lower_tokens if t in contrast_words),
        # --- Sarcasm  ---
            "sarcasm_markers":   len(sarcasm_pattern.findall(original_text)),
                
            "intensity_count":   sum(1 for t in lower_tokens if t in intensity_words),
            "exclamation_count": len(re.findall(r'(?<!!)!(?!!)', original_text)),

        # --- syntactic complexity  ---
            "dep_depth":        max((len(list(t.ancestors)) for t in doc), default=0),
            "clause_count":     sum(1 for t in doc if t.dep_ in {"ccomp", "advcl", "relcl"}),
            "modal_count":      sum(1 for t in doc if t.tag_ == "MD"),  # should, would, must

        # --- lexical richness ---
            "type_token_ratio": len(set(lower_tokens)) / n,   # lexical diversity
            "is_question":      int(doc[-1].text == "?") if len(doc) > 0 else 0,

        # --- Sentiment --- ---
            "polarity":         TextBlob(original_text).sentiment.polarity,
            
        # --- Emotion complexity (VADER) ---
        # emotion_diversity:  are both positive AND negative tones present?
        # High value = emotionally mixed = more likely to cause disagreement
            "emotion_diversity":  int(vs['pos'] > 0) + int(vs['neg'] > 0),

        # emotion_intensity:  how strong is the overall emotion?
        # High absolute value = strong clear emotion
        # Low value = weak/neutral = harder to label = more disagreement
            "emotion_intensity":  abs(vs['compound']),

        # emotional_conflict: are BOTH positive and negative meaningfully present?
        # True = sentence pulls in two directions = annotators will split
            "emotional_conflict": int(vs['pos'] > 0.1 and vs['neg'] > 0.1),
    }

# Batch processing — same as before, no change needed here
texts   = df["text"].astype(str).tolist()
records = []
for doc, original_text in zip(nlp.pipe(texts, batch_size=512, n_process=1), texts):
    records.append(extract_features_from_doc(doc, original_text))

features = pd.DataFrame(records)
df_final = pd.concat([df, features], axis=1)

PROC_DIR.mkdir(parents=True, exist_ok=True)
df_final.to_csv(FEATURES_FILE, index=False)
print("Done!", df_final.shape)