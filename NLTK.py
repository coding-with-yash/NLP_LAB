import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.chunk import tree2conlltags

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


raw_text=("The Board of Control for Cricket in India (BCCI) is the governing body for cricket in India"
          " and is under the jurisdiction of Ministry of Youth Affairs and Sports, Government of India. "
          "[2] The board was formed in December 1928 as a society, registered under the Tamil Nadu Societies "
          "Registration Act. It is a consortium of state cricket associations and the state associations select "
          "their representatives who in turn elect the BCCI Chief. Its headquarters are in Wankhede Stadium, Mumbai."
          "Grant Govan was its first president and Anthony De Mello its first secretary.")


raw_words = word_tokenize(raw_text)         # Tokenize the input text

tags = pos_tag(raw_words)                   # Perform part-of-speech tagging on the words

ne = ne_chunk(tags, binary=True)            # Apply named entity recognition

iob = tree2conlltags(ne)                    # Convert the named entity chunks to IOB format

for word, pos, tag in iob:                  # Print the IOB tags
    print(word, pos, tag)
