from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from fastapi.responses import StreamingResponse
import json
from collections import defaultdict
from pinecone import Pinecone
from lxml import etree
import unicodedata
import uvicorn
from fastapi.responses import JSONResponse
from openai import OpenAI

from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = OpenAI()

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "amaicus"
index = pc.Index(index_name)

# Initialize Pinecone Vector Store with LangChain
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize FastAPI
app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Read the doc_id to filename mapping from the JSON file
with open('doc_id_mapping.json', 'r') as f:
    doc_id_to_filename = json.load(f)

# Define the SearchRequest model
class SearchRequest(BaseModel):
    query: str
    location: str = ""  # Optional location code
    selected_type: str = "Any"  # Optional document type

@app.post("/search")
def search_documents(search_request: SearchRequest):
    query = search_request.query
    location = search_request.location
    selected_type = search_request.selected_type

    print(f"Received search request with query: {query}")
    print(f"Location: {location}, Document type: {selected_type}")

    # Perform the search
    results = handle_search(query, selected_location_code=location, selected_type=selected_type)

    print(f"Processing {len(results)} search results...")

    # Process the search results and return top 50 results grouped by document
    processed_results = process_search_results(results)

    print(f"Processed {len(processed_results)} documents.")
    for doc in processed_results:
        print(f"Document ID: {doc['doc_id']} has {len(doc['search_hits'])} search hits.")

    # Return the processed results to the frontend as JSON
    return {"documents": processed_results}

def handle_search(query, selected_location_code="", selected_type="Any"):
    """
    Handles the search by modifying the query, searching through Pinecone,
    and returning the results.
    """
    modified_query = query

    # Construct metadata filter for Pinecone if any filters are applicable
    filters = {}
    if selected_location_code:
        filters["country"] = selected_location_code
    if selected_type != "Any" and selected_type != "Similarity Score":  # Do not add "Similarity Score" as a filter
        filters["subtype"] = selected_type

    print(f"Querying Pinecone with query: {modified_query} and filters: {filters}")

    # Query Pinecone with the modified query
    results = vector_store.similarity_search_with_score(
        query=modified_query, 
        k=50,  # Retrieve more results to ensure we have enough to select top 5 per document
        filter=filters if filters else None
    )

    return results

def process_search_results(results):
    """
    Processes the search results, groups them by doc_id, 
    and keeps only the top 5 search results per document based on similarity score.
    Within these top 5, results are sorted by chunk_id.
    """
    grouped_results = defaultdict(list)

    # Cache for parsed XML documents to avoid re-parsing the same file multiple times
    xml_cache = {}

    # Group results by doc_id
    for result, score in results:
        doc_id = result.metadata.get("doc_id", "")
        chunk_id = result.metadata.get("chunk_id", "")
        eId = result.metadata.get("eId", "")
        tag = result.metadata.get("tag", "")
        country = result.metadata.get("country", "")
        date = result.metadata.get("date", "")
        subtype = result.metadata.get("subtype", "")
        expression_date = result.metadata.get("expression_date", "")
        parents = result.metadata.get("parents", [])  # List of JSON strings

        # Get the filename from doc_id
        filename = doc_id_to_filename.get(doc_id, "")
        if not filename:
            print(f"Filename not found for doc_id: {doc_id}")
            continue

        # Load and parse the XML file if not already cached
        if filename not in xml_cache:
            xml_file_path = os.path.join('expressions', filename)
            if not os.path.exists(xml_file_path):
                print(f"XML file not found: {xml_file_path}")
                continue
            try:
                parser = etree.XMLParser(remove_blank_text=True)
                tree = etree.parse(xml_file_path, parser)
                root = tree.getroot()
                nsmap = {"a": root.nsmap[None]}
                xml_cache[filename] = (root, nsmap)
            except Exception as e:
                print(f"Error parsing XML file {xml_file_path}: {e}")
                continue
        else:
            root, nsmap = xml_cache[filename]

        # Function to extract fields from an element
        def extract_fields(element):
            heading = extract_component_text(element, "./a:heading", nsmap)
            subheading = extract_component_text(element, "./a:subheading", nsmap)
            crossheading = extract_component_text(element, "./a:crossHeading", nsmap)
            intro = extract_component_text(element, "./a:intro", nsmap)
            num = extract_component_text(element, "./a:num", nsmap, is_text_node=True)
            return heading, subheading, crossheading, intro, num

        # Find the element corresponding to the eId
        element = root.xpath(f"//*[@eId='{eId}']", namespaces=nsmap)
        if not element:
            print(f"Element with eId {eId} not found in {filename}")
            continue
        else:
            element = element[0]

        # Extract fields for the hit component
        heading, subheading, crossheading, intro, num = extract_fields(element)

        # Process parents
        processed_parents = []
        for parent_json in parents:
            parent_dict = json.loads(parent_json)
            parent_eId = parent_dict.get('eId', '')
            parent_tag = parent_dict.get('tag', '')

            # Find the parent element
            parent_element = root.xpath(f"//*[@eId='{parent_eId}']", namespaces=nsmap)
            if not parent_element:
                print(f"Parent element with eId {parent_eId} not found in {filename}")
                continue
            else:
                parent_element = parent_element[0]

            # Extract fields for the parent
            p_heading, p_subheading, p_crossheading, p_intro, p_num = extract_fields(parent_element)

            processed_parents.append({
                "tag": parent_tag,
                "eId": parent_eId,
                "heading": p_heading,
                "subheading": p_subheading,
                "crossheading": p_crossheading,
                "intro": p_intro,
                "num": p_num
            })

        # Build the search hit
        # Extract only the last paragraph of the content
        content = result.page_content.strip()
        last_paragraph = extract_last_paragraph(content)

        # Ensure chunk_id is integer for sorting
        try:
            chunk_id_int = int(chunk_id)
        except ValueError:
            chunk_id_int = 0  # Default to 0 if conversion fails

        search_hit = {
            "alias": "",  # Will set alias at the document level
            "subtype": subtype,
            "country": country,
            "date": date,
            "expression_date": expression_date,
            "score": score,
            "heading": heading,
            "subheading": subheading,
            "crossheading": crossheading,
            "intro": intro,
            "num": num,
            "tag": tag,
            "eId": eId,
            "content": last_paragraph,  # Use only the last paragraph
            "doc_id": doc_id,
            "chunk_id": chunk_id_int,  # Use integer chunk_id
            "parents": processed_parents,
        }

        grouped_results[doc_id].append(search_hit)

    # Debugging: Show grouped results per doc_id
    print(f"Grouped results by doc_id: {len(grouped_results)} documents found.")

    processed_results = []

    # For each document, process the top 5 hits
    for doc_id, doc_results in grouped_results.items():
        # Sort by similarity score in descending order
        sorted_by_score = sorted(doc_results, key=lambda x: x["score"], reverse=True)
        # Keep only top 5 results per document
        top_5_results = sorted_by_score[:5]
        # Now sort these top 5 results by chunk_id in ascending order
        top_5_sorted_by_chunk = sorted(top_5_results, key=lambda x: x["chunk_id"])

        # Since we have the XML root for this doc_id, extract alias from FRBRWork
        filename = doc_id_to_filename.get(doc_id, "")
        if filename in xml_cache:
            root, nsmap = xml_cache[filename]
            frbr_alias = root.xpath(".//a:FRBRWork/a:FRBRalias/@value", namespaces=nsmap)
            alias = frbr_alias[0] if frbr_alias else ""
        else:
            alias = ""

        # Structure the response with document-level info and search hits
        processed_results.append({
            "doc_id": doc_id,
            "alias": alias,  # Use alias extracted from XML
            "subtype": top_5_sorted_by_chunk[0]["subtype"],
            "country": top_5_sorted_by_chunk[0]["country"],
            "date": top_5_sorted_by_chunk[0]["date"],
            "expression_date": top_5_sorted_by_chunk[0]["expression_date"],
            "search_hits": top_5_sorted_by_chunk  # Include top 5 search results per document, sorted by chunk_id
        })

    return processed_results

# Helper function to extract last paragraph
def extract_last_paragraph(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if paragraphs:
        return paragraphs[-1]
    else:
        return text  # Or return empty string

# Helper functions provided by you

def parse_xml(file_path):
    # Parse the XML file directly to preserve line numbers
    parser = etree.XMLParser()
    tree = etree.parse(file_path, parser)
    root = tree.getroot()
    return root

# List of valid components including attachment
valid_components = ["section", "chapter", "part", "subsection", "article", 
                    "preface", "preamble", "title", "subparagraph", 
                    "paragraph", "clause", "item", "subitem", "p", "attachment"]

# Function to clean and normalize text
def clean_text(text):
    # Normalize Unicode to remove accents and convert to simple ASCII where possible
    normalized_text = unicodedata.normalize('NFKC', text)
    
    # Replace common Unicode punctuation with their ASCII equivalents
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote (apostrophe)
        '\u201C': '"',  # Left double quote
        '\u201D': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
    }
    
    for unicode_char, ascii_char in replacements.items():
        normalized_text = normalized_text.replace(unicode_char, ascii_char)

    # Replace unwanted characters like newlines, tabs, etc.
    cleaned_text = ' '.join(normalized_text.split())

    return cleaned_text

def extract_filtered_text(element, nsmap):
    """
    Extracts text from the current element excluding headings, subheadings, crossheadings, intros, and remarks.
    """
    # Exclude the specified tags
    text = ' '.join(element.xpath(
        ".//text()[not(ancestor::a:heading) and not(ancestor::a:subheading) "
        "and not(ancestor::a:crossHeading) and not(ancestor::a:intro) "
        "and not(ancestor::a:remark)]", namespaces=nsmap))
    
    return clean_text(text) if text else "N/A"

# Function to extract all non-remark text from an element
def extract_non_remark_text(element, nsmap):
    # Extract text and join all parts
    text = ' '.join(element.xpath(".//text()[not(ancestor::a:remark)]", namespaces=nsmap))
    # Clean the extracted text
    return clean_text(text) if text else "N/A"

# Helper function to extract text content with fallback to "N/A"
def extract_component_text(element, xpath_expr, nsmap, is_text_node=False):
    """
    Extracts the content of the given element based on the XPath expression.
    If is_text_node is True, it handles direct text extraction.
    """
    extracted_elements = element.xpath(xpath_expr, namespaces=nsmap)
    
    if is_text_node:
        # For text nodes, return the text directly
        if extracted_elements:
            extracted_element = extracted_elements[0]
            if isinstance(extracted_element, etree._Element):
                # Get text content of the element
                text_content = extracted_element.text
                if text_content:
                    return clean_text(text_content.strip())
                else:
                    return "N/A"
            elif isinstance(extracted_element, str):
                return clean_text(extracted_element.strip())
            else:
                return "N/A"
        else:
            return "N/A"
    
    # For element nodes, apply the usual non-remark text extraction
    if extracted_elements:
        return extract_non_remark_text(extracted_elements[0], nsmap)
    else:
        return "N/A"
    
# # Serve static files (if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Add GZip middleware to compress responses
app.add_middleware(GZipMiddleware, minimum_size=1000)
    
def traverse_xml_for_toc(element, nsmap, parents=None, depth=0):
    valid_components = [
        "section", "chapter", "part", "subsection", "article", "preface",
        "preamble", "title", "subparagraph", "paragraph", "clause", "item",
        "subitem", "p", "attachment"
    ]

    if parents is None:
        parents = []

    tag_name = etree.QName(element).localname

    # Special case: Handle <preface> without heading or eId
    if tag_name == "preface":
        current_toc_entry = {
            "title": "Preface",  # Manually set the title as "Preface"
            "id": "",  # No eId for preface
            "level": depth,
            "children": []
        }

        # Traverse children of preface (if any)
        for child in element:
            child_toc = traverse_xml_for_toc(child, nsmap, parents + [current_toc_entry], depth + 1)
            if child_toc:
                current_toc_entry["children"].extend(child_toc)

        return [current_toc_entry]

    # Process other valid components
    if tag_name in valid_components:
        eId = element.get("eId", "")
        heading_text = extract_component_text(element, "./akn:heading", nsmap)
        subheading_text = extract_component_text(element, "./akn:subheading", nsmap)
        crossheading_text = extract_component_text(element, "./akn:crossHeading", nsmap)
        num_text = extract_component_text(element, "./akn:num/text()", nsmap, is_text_node=True)

        if not heading_text and not subheading_text and not crossheading_text and tag_name != "preface":
            return []

        title_text = f"{tag_name.capitalize()} {num_text}: {heading_text or subheading_text or crossheading_text}".strip()
        current_toc_entry = {
            "title": title_text,
            "id": eId,
            "level": depth,
            "children": []
        }

        updated_parents = parents + [current_toc_entry]

        for child in element:
            child_toc = traverse_xml_for_toc(child, nsmap, updated_parents, depth + 1)
            if child_toc:
                current_toc_entry["children"].extend(child_toc)

        return [current_toc_entry]

    else:
        toc_items = []
        for child in element:
            child_toc = traverse_xml_for_toc(child, nsmap, parents, depth)
            if child_toc:
                toc_items.extend(child_toc)
        return toc_items

def extract_component_text(element, xpath_expr, nsmap, is_text_node=False):
    extracted_elements = element.xpath(xpath_expr, namespaces=nsmap)
    
    if not extracted_elements:
        return ""

    if is_text_node:
        return extracted_elements[0].strip() if isinstance(extracted_elements[0], str) else extracted_elements[0].text.strip()

    if isinstance(extracted_elements[0], etree._Element):
        return "".join(extracted_elements[0].itertext()).strip()

    return extracted_elements[0].strip()

def parse_xml_to_toc(xml_content):
    try:
        root = etree.fromstring(xml_content.encode('utf-8'))
        nsmap = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}

        toc_items = traverse_xml_for_toc(root, nsmap)

        if not toc_items:
            print("No TOC items generated.")
        return toc_items

    except Exception as e:
        print(f"Error parsing XML: {e}")
        return []


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    """
    Returns the HTML content, XML content, and TOC (Table of Contents) for the document with the given doc_id.
    """
    # Get the filename from doc_id
    filename = doc_id_to_filename.get(doc_id, "")
    if not filename:
        return JSONResponse(content={"error": f"Filename not found for doc_id: {doc_id}"}, status_code=404)

    # Replace the extension with .html
    html_filename = filename.rsplit('.', 1)[0] + '.html'

    # Paths to the XML and HTML files
    xml_file_path = os.path.join('expressions', filename)
    html_file_path = os.path.join('expressions_html', html_filename)

    # Check if files exist
    if not os.path.exists(xml_file_path):
        return JSONResponse(content={"error": f"XML file not found: {xml_file_path}"}, status_code=404)
    if not os.path.exists(html_file_path):
        return JSONResponse(content={"error": f"HTML file not found: {html_file_path}"}, status_code=404)

    try:
        # Load and parse the XML file to extract TOC
        with open(xml_file_path, 'r', encoding='utf-8') as f_xml:
            xml_content = f_xml.read()

        # Generate the TOC using the XML content
        toc = parse_xml_to_toc(xml_content)

        # Load the HTML content
        with open(html_file_path, 'r', encoding='utf-8') as f_html:
            html_content = f_html.read()

        # Return the HTML content, XML content, TOC, and FRBR expression URI
        return JSONResponse(content={
            "html_content": html_content,
            "xml_content": xml_content,
            "toc": toc
        })
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing document {doc_id}: {e}"}, status_code=500)
    
# Model to receive the summarization request
class SummarizeRequest(BaseModel):
    eId: str
    docId: str

@app.post("/summarize")
async def summarize_section(request: SummarizeRequest):
    eId = request.eId
    doc_id = request.docId

    # Get the filename from the doc_id
    filename = doc_id_to_filename.get(doc_id, "")
    if not filename:
        return {"error": f"Filename not found for doc_id: {doc_id}"}

    xml_file_path = os.path.join('expressions', filename)
    if not os.path.exists(xml_file_path):
        return {"error": f"XML file not found for doc_id: {doc_id}"}

    # Parse the XML file
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(xml_file_path, parser)
        root = tree.getroot()
        nsmap = {"akn": root.nsmap[None]}  # Use namespace

        # Find the element corresponding to the eId
        element = root.xpath(f"//*[@eId='{eId}']", namespaces=nsmap)
        if not element:
            return {"error": f"Element with eId {eId} not found in {filename}"}

        element = element[0]
        section_text = " ".join(element.xpath(".//text()")).strip()

    except Exception as e:
        return {"error": f"Error processing XML: {str(e)}"}
                
    def stream_openai_response():
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant for legal research."},
                    {"role": "user", "content": f"Regarding the following section from a legal document, briefly outline the overall context and relevance of the section. Our goal is to provide the user with an understanding of whether the section is relevant for their research or not. Your response should be a single paragraph of no more than 100 words. Here is the section: {section_text}"}
                ],
                stream=True  # Enable streaming
            )

            # Debug: Track how many chunks we are receiving from OpenAI
            chunk_count = 0

            for chunk in stream:
                # Check if chunk has content
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(f"Chunk {chunk_count}: {content}")  # Debugging: Print each chunk
                    chunk_count += 1
                    yield content

            # Debug: Print total chunks received
            print(f"Total chunks received: {chunk_count}")

        except Exception as e:
            print(f"Error in streaming OpenAI response: {e}")
            yield "Error in streaming OpenAI response."


    return StreamingResponse(stream_openai_response(), media_type="text/event-stream")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use the PORT environment variable set by Railway
    uvicorn.run(
        "main:app",  # Assuming your FastAPI instance is named `app` in a file called `main.py`
        host="0.0.0.0",
        port=port,
        loop="asyncio",  # Use asyncio loop (optional, but may help with streaming)
        http="h11",  # Use HTTP/1.1
    )
