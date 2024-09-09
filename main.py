# /**********************************************************************************************************
# Step 1: Import necessary libraries
# /**********************************************************************************************************
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from nltk.stem import WordNetLemmatizer
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from functions import preprocess_name, find_similar_names

# Load in yaml file for environment level variables
load_dotenv('env')

# Access the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Step 2: Set up OpenAI API key
model = OpenAIEmbeddings(model='text-embedding-ada-002')
llm = ChatOpenAI(temperature=0.0)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# /**********************************************************************************************************
# Step 2: Find similar names within the events themselves
# /**********************************************************************************************************
text = """
I recently visited Greenwood Health Centre and had an outstanding experience!

First off, the service was exceptional. Dr. Alice Smith greeted us with a warm smile and made sure we were comfortable. Nurse Bob Jonson was incredibly attentive and friendly. He even recommended some helpful tips for managing my condition.

The care I received was fantastic! Dr. Charlie Brown is very knowledgeable, and you can tell he puts a lot of effort into his consultations. He explained everything clearly and made sure I understood my treatment plan. I also had the pleasure of meeting Alice again when she came by to check on me during my follow-up visit.

I can’t forget to mention the support staff. Nurse David Lee and Nurse Eve Williams have done a wonderful job ensuring the clinic runs smoothly. It’s clean and inviting, with a perfect blend of professionalism and comfort. Eve even took the time to chat with us about the clinic’s services and their commitment to patient care.

Overall, Greenwood Health Centre is a gem. Whether you’re there for a routine check-up or more specialized care, you’re in good hands. Kudos to Dr. Smith, Nurse Bob Jonson, Dr. Charlie Brown, David Lee, and Eve Williams for making my visit so memorable. I’ll definitely be back!

Feel free to use this review to test your embeddings! If you need any more examples or further assistance, just let me know.
"""

# Set prompt template and defined chain
prompt = ChatPromptTemplate(
    [
        ("system", "You are an AI developed to identify individuals mentioned in a text and find similar names within the text given the context"),
        ("user", "Instructions are to find the names mentioned in the complaint and identify other mentions of the individual. The response should be in json format. The output should have a base name as a key and all the similar names as values."),
        ("assistant", "Understood, are there any constraints"),
        ("user", "Yes. Only identify names that are mentioned in the complaint and provide similar names within the text if you are confident. There could be more than 1 similar name for each base name. Reflect on the context that names are mentioned. If you are unsure, do not provide any similar names. The base name should be the most complete name."),
        ("assistant", "Do you have any specific examples to test the model with?"),
        ("user", "Yes. The output could look like this: 'Dr. Oliver Armstrong': ['Oliver', 'Dr. Armstrong', 'Oliver Armstrong']"),
        ("assistant", "Great! Let's get started. Please provide the text you would like to analyze."),
        ("user", "Text: {text}"),
    ]
)

# Set up a chain to process the prompt
chain = prompt | llm 
response = chain.invoke(text)
print(response.content)

# /**********************************************************************************************************
# Step 3: Find similar names across events
# /**********************************************************************************************************
names = ["Dr. Alice Smith", "Nurse Bob Jonson", "Dr. Charlie Brown", "Alice", "Nurse David Lee", "Nurse Eve Williams", "Dr. Smith", "Bob Jonson", "Charlie Brown", "David Lee", "Eve Williams"]
preprocessed_names = [preprocess_name(name, lemmatizer) for name in names]
similar_names = find_similar_names(preprocessed_names, model, threshold=0.85)
print(similar_names)
