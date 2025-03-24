import re

def clean_text(text):
    # Define allowed characters: English letters, numbers, punctuation, common symbols, and whitespace
    allowed_chars = r"[^a-zA-Z0-9.,!?;:'\"()\[\]{}\-+=_/\\*&^%$#@<>’| \n”“•]"

    # Remove unwanted characters while keeping spaces and newlines
    return re.sub(allowed_chars, "~", text)

# Example Usage:
#sample_text = """The Independent Jane
#"bite" 'bite' 
#For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence.
#Elizabeth’s refusal of Mr. Collins -- offer of marriage showed an | independence seldom seen in heroines of the day.
#Her refusal of Mr. Darcy while triggered by anger showed a level of independence that left him shocked and stunned.
#こんにちは！(Hello in Japanese) should be removed.
#"""

f = open("output2.txt", "r")
sample_text = f.read()
f.close()

f = open("nice_output2.txt", "w")
f.write(clean_text(sample_text))
f.close()