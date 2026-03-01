import nltk
import ssl

# Bypass SSL verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK data...")

# Download all required datasets
nltk.download('punkt', quiet=False)
nltk.download('punkt_tab', quiet=False)
nltk.download('stopwords', quiet=False)
nltk.download('wordnet', quiet=False)

print("✅ All NLTK data downloaded successfully!")