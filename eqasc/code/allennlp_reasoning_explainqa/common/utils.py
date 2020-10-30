import string

def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s.strip()
