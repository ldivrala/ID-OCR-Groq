!sudo apt-get update
!sudo apt-get install -y tesseract-ocr
!pip install pytesseract opencv-python-headless numpy python-dotenv

!pip install requests langchain_community langchain_groq PyPDF2 pydantic opencv-python


import pytesseract
import cv2
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
# Constants
IMAGE_PATH = 'sample_data/id_.jpeg'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define template for extracting ID details
TEMPLATE = """
Extract and format the details into JSON from the following text:

Example 1:
[
  "Text": “>. ‘74
     ~ a: ARAB EMIRATES a3 SRS ela Eee
     FEDERAL AUTHORITY FOR iDEN? eS Delle)
     COAZENSHIP CUSTOMS & PORT CUR saetne at tah,
     “ Redlaent Identity Care “ a! j we ye Ma
     ‘%.
     10 Number / 4, 545) at, \
     884-1992.  5887987-8 i
     Se ee A pe es | aly ay!
     Name Mahondra Shekhawat D epemdra Shekhawat
     Data of Berth 03/06/1990 ll ALS
     Se
     Nevenatity india
     teswing Date (27) > see
     borhan 07/07/2027 3S: ual!
     Signayure Sew Expirr Date /+420) & Sex: N
     06/07/2028
  ”,
  "Output": [
    "Name": "Mahendra Shekhawat Dependra Shekhawat",
    "ID_number": "884-1992-5887987-8",
    "Date_of_birth": "03/05/1990",
    "Sex": "M",
    "Issuing_Date": "07/07/2027",
    "Expiry_Date": "06/07/2028"
  ]
]

Example 2:
[
  "Text": “>. ‘67
     ~ a: EXAMPLE COUNTRY a3 SRS exa Eee
     NATIONAL AUTHORITY FOR iDEN? eS Delle)
     CITIZENSHIP CUSTOMS & PORT CUR saetne at tah,
     “ Resident Identity Card “ a! j we ye Ex
     ‘%.
     10 Number / 4, 333) at, \
     999- 8888-  1234567-9 i
     Se ee A pe es | aly ay!
     Name Alice Johnson
     Date of Berth 11/11/1985 ll ALS
     Se
     Nevenatity example
     teswing Date (27) > see
     borhan 01/01/2025 3S: ual!
     Signature See Expiry Date /+420) & Sox: F
     01/01/2035
  ”,
  "Output": [
    "Name": "Alice Johnson",
    "ID_number": "999-8888-1234567-9",
    "Date_of_birth": "11/11/1985",
    "Sex": "F",
    "Issuing_Date": "01/01/2025",
    "Expiry_Date": "01/01/2035"
  ]
]

Text:
{prompt}

Format the output as JSON Object:
"Name": "str",
"ID_number": "str",
"Date_of_birth": "str",
"Sex": "str",
"Issuing_Date": "str",
"Expiry_Date": "str"

Provide the output with the details in the above JSON Object format.
"""

# Define the ID details model
class IdDetail(BaseModel):
    Name: str
    ID_number: str
    Date_of_birth: str
    Sex: str
    Issuing_Date: str
    Expiry_Date: str

# Initialize OCR and image processing
def extract_text_from_image(image_path: str) -> str:
    # Read the image from file
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply a slight blur to reduce noise and improve OCR accuracy
    img = cv2.medianBlur(img, 3)
    
    # Configure pytesseract for optimal ID text extraction
    custom_config = r'--oem 3 --psm 6'
    
    # Extract text from the processed image
    text = pytesseract.image_to_string(img, config=custom_config)
    return text

# Extract text from the image
text = extract_text_from_image(IMAGE_PATH)

# Initialize the Groq API model and parser
llm = ChatGroq(model_name='llama3-70b-8192', groq_api_key=GROQ_API_KEY)
parser = PydanticOutputParser(pydantic_object=IdDetail)
slide_content_prompt = ChatPromptTemplate.from_template(TEMPLATE)

# Define the chain to invoke the model
def parse_id_details(text: str) -> IdDetail:
    result = (
        {"prompt": lambda x: x["prompt"]}
        | slide_content_prompt
        | llm
        | parser
    ).invoke({"prompt": text})
    return result

# Invoke the chain and get the result
id_details = parse_id_details(text)

# Output the result
print(id_details.json())
