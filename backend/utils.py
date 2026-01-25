from typing import List, Dict, Optional
import io
import re
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import logging
import hashlib
from datetime import datetime
from PyPDF2.generic import IndirectObject


logger = logging.getLogger(__name__)

# Initialize LLM for extraction with retry logic
def get_llm():
    """Get LLM instance with configuration"""
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_retries=3,
        request_timeout=60
    )

def process_document(content: bytes, filename: str, file_id: str) -> List[Dict]:
    """
    Process document with advanced extraction and chunking
    
    Args:
        content: Raw file bytes
        filename: Original filename
        file_id: Unique identifier for the file
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    
    logger.info("="*80)
    logger.info(f"📄 Processing document: {filename}")
    logger.info(f"File ID: {file_id}")
    logger.info(f"File size: {len(content)} bytes")
    
    try:
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text, extraction_metadata = extract_pdf_text(content, filename)
        elif filename.endswith(('.docx', '.doc')):
            text, extraction_metadata = extract_docx_text(content, filename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        # Validate extracted text
        if not text or len(text.strip()) < 100:
            raise ValueError(f"Insufficient text extracted from {filename} (only {len(text)} chars)")
        
        logger.info(f"✓ Extracted text: {len(text)} characters")
        logger.info(f"✓ Word count: {len(text.split())} words")
        logger.info(f"✓ Preview: {text[:300]}...")
        
        # Generate document hash for deduplication
        doc_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        logger.info(f"✓ Document hash: {doc_hash}")
        
        # Extract structured information using LLM
        logger.info("\n🤖 Starting LLM-based structured extraction...")
        structured_summary = extract_with_llm(text, filename, extraction_metadata)
        
        if structured_summary:
            logger.info(f"✅ Structured summary created: {len(structured_summary)} characters")
            logger.info(f"Summary preview:\n{structured_summary[:600]}...\n")
        else:
            logger.warning("⚠️  No structured summary created - continuing with basic chunks")
        
        # Create smart chunks
        logger.info("📦 Creating document chunks...")
        chunks = create_smart_chunks(text, filename, file_id, extraction_metadata)
        logger.info(f"✓ Created {len(chunks)} content chunks")
        
        # Add structured summary as first chunk (highest priority)
        if structured_summary:
            summary_chunk = {
                "text": structured_summary,
                "metadata": {
                    "source": filename,
                    "file_id": file_id,
                    "doc_hash": doc_hash,
                    "chunk_id": -1,
                    "chunk_type": "structured_summary",
                    "priority": "high",
                    "created_at": datetime.now().isoformat(),
                    "chunk_count": len(chunks),
                    **extraction_metadata
                }
            }
            chunks.insert(0, summary_chunk)
            logger.info("✅ Inserted structured summary as priority chunk")
        
        logger.info(f"✓ Total chunks ready for storage: {len(chunks)}")
        logger.info(f"✓ Processing completed successfully")
        logger.info("="*80 + "\n")
        
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Error processing document {filename}: {e}", exc_info=True)
        raise

def extract_pdf_text(content: bytes, filename: str) -> tuple[str, Dict]:
    try:
        logger.info("  Extracting from PDF...")
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)

        num_pages = len(pdf_reader.pages)
        logger.info(f"  PDF has {num_pages} pages")

        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num} ---\n{page_text}\n"

        pdf_metadata = pdf_reader.metadata or {}

        metadata = {
            "document_type": "pdf",
            "page_count": num_pages,
            "pdf_title": str(pdf_metadata.get("/Title", "")),
            "pdf_author": str(pdf_metadata.get("/Author", "")),
            "has_images": pdf_has_images(pdf_reader),
        }

        logger.info(f"  ✓ Extracted {len(text)} chars from {num_pages} pages")
        return text.strip(), metadata

    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise ValueError(f"Failed to extract text from PDF: {e}")

    
def pdf_has_images(pdf_reader):
    """Safely check if PDF has images, handling IndirectObject errors gracefully."""
    try:
        for page in pdf_reader.pages:
            # Safely get resources - handle IndirectObject
            try:
                resources = page.get("/Resources")
                if not resources:
                    continue
                # Resolve if IndirectObject
                if hasattr(resources, 'get_object'):
                    resources = resources.get_object()
            except:
                continue

            # Safely get XObject
            try:
                xobject = resources.get("/XObject")
                if not xobject:
                    continue
                # Resolve if IndirectObject
                if hasattr(xobject, 'get_object'):
                    xobject = xobject.get_object()
            except:
                continue

            # Check each XObject safely
            try:
                if isinstance(xobject, dict):
                    for obj_key, obj in xobject.items():
                        try:
                            # Double resolve for nested IndirectObjects
                            obj_resolved = obj
                            if hasattr(obj, 'get_object'):
                                obj_resolved = obj.get_object()
                            
                            # Check if it's an image
                            if isinstance(obj_resolved, dict):
                                subtype = obj_resolved.get("/Subtype")
                                if subtype == "/Image":
                                    return True
                        except:
                            # Skip problematic objects
                            continue
            except:
                continue
                
    except Exception:
        # If anything fails, assume no images to avoid blocking text extraction
        pass
    
    return False

def extract_docx_text(content: bytes, filename: str) -> tuple[str, Dict]:
    """
    Extract text from DOCX with ENHANCED TABLE HANDLING for better retrieval
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    try:
        logger.info("  Extracting from DOCX with enhanced table processing...")
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        
        text = ""
        table_count = 0
        paragraph_count = 0
        
        # Track document elements in order (paragraphs and tables)
        for element in doc.element.body:
            # Handle paragraphs
            if element.tag.endswith('p'):
                para_text = element.text.strip() if hasattr(element, 'text') else ''
                if para_text:
                    text += para_text + "\n"
                    paragraph_count += 1
            
            # Handle tables
            elif element.tag.endswith('tbl'):
                table_count += 1
                
                # Get the actual table object
                table = None
                for tbl in doc.tables:
                    if tbl._element == element:
                        table = tbl
                        break
                
                if table:
                    logger.info(f"    Processing table {table_count}: {len(table.rows)} rows x {len(table.columns)} cols")
                    
                    # Extract table with intelligent formatting
                    table_text = extract_table_intelligently(table, table_count)
                    text += table_text
        
        metadata = {
            "document_type": "docx",
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "has_tables": table_count > 0,
        }
        
        logger.info(f"  ✓ Extracted {paragraph_count} paragraphs, {table_count} tables")
        return text.strip(), metadata
        
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise ValueError(f"Failed to extract text from DOCX: {e}")

def extract_table_intelligently(table, table_number: int) -> str:
    """
    Extract table with intelligent formatting for retrieval
    Converts tables into natural language for better semantic search
    """
    
    table_text = f"\n{'='*60}\n"
    table_text += f"TABLE {table_number} START\n"
    table_text += f"{'='*60}\n\n"
    
    # Detect table type
    num_cols = len(table.columns)
    num_rows = len(table.rows)
    
    # Extract all cells
    table_data = []
    for row in table.rows:
        row_data = [cell.text.strip() for cell in row.cells]
        table_data.append(row_data)
    
    # Check if it's a 2-column key-value table
    if num_cols == 2:
        logger.info(f"      Detected 2-column key-value table")
        table_text += process_key_value_table(table_data, table_number)
    
    # Check if it's a multi-column data table
    elif num_cols > 2:
        logger.info(f"      Detected {num_cols}-column data table")
        table_text += process_multi_column_table(table_data, table_number)
    
    else:
        # Single column - just list items
        for row_data in table_data:
            if row_data[0]:
                table_text += f"- {row_data[0]}\n"
    
    table_text += f"\n{'='*60}\n"
    table_text += f"TABLE {table_number} END\n"
    table_text += f"{'='*60}\n\n"
    
    return table_text

def process_key_value_table(table_data: List[List[str]], table_num: int) -> str:
    """
    Process 2-column key-value tables (like Key Metrics)
    Converts to natural language for better retrieval
    """
    
    text = ""
    
    # Check if first row is header
    first_row = table_data[0] if table_data else []
    has_header = (first_row and 
                  any(kw in first_row[0].lower() for kw in ['metric', 'key', 'item', 'field', 'name']))
    
    if has_header:
        text += f"Table: {first_row[0]} and {first_row[1]}\n\n"
        data_rows = table_data[1:]
    else:
        data_rows = table_data
    
    # Process each key-value pair
    for row_data in data_rows:
        if len(row_data) >= 2 and row_data[0] and row_data[1]:
            key = row_data[0].strip()
            value = row_data[1].strip()
            
            # Skip if it's just separators or empty
            if not key or not value or key == '---' or value == '---':
                continue
            
            # Format in multiple ways for better retrieval
            text += f"📊 {key}: {value}\n"
            text += f"The {key} is {value}.\n"
            text += f"{key} = {value}\n"
            
            # Add alternative phrasings
            key_lower = key.lower()
            if 'count' in key_lower or 'number' in key_lower:
                text += f"There are {value} {key.lower().replace('number of', '').replace('count', '').strip()}.\n"
            
            text += "\n"
    
    return text

def process_multi_column_table(table_data: List[List[str]], table_num: int) -> str:
    """
    Process multi-column tables (like branch locations)
    """
    
    text = ""
    
    if not table_data:
        return text
    
    # First row is typically header
    headers = table_data[0]
    data_rows = table_data[1:]
    
    text += f"Table Headers: {' | '.join(headers)}\n\n"
    
    # Process each data row
    for idx, row_data in enumerate(data_rows, 1):
        if not any(row_data):  # Skip empty rows
            continue
        
        text += f"Entry {idx}:\n"
        
        # Create key-value pairs from headers and row data
        for header, value in zip(headers, row_data):
            if header and value:
                text += f"  {header}: {value}\n"
                text += f"  The {header.lower()} is {value}.\n"
        
        # Also create a plain text version
        text += f"  " + " | ".join([v for v in row_data if v]) + "\n\n"
    
    return text

def extract_with_llm(text: str, filename: str, doc_metadata: Dict) -> Optional[str]:
    """
    Use LLM to extract comprehensive structured information with TABLE FOCUS
    
    Args:
        text: Full document text
        filename: Document filename
        doc_metadata: Document metadata from extraction
        
    Returns:
        Structured summary string or None if extraction fails
    """
    
    # Determine optimal sample size
    if len(text) > 15000:
        sample_text = text[:15000]
        logger.info(f"  Using first 15,000 characters for extraction (doc is {len(text)} chars)")
    else:
        sample_text = text
        logger.info(f"  Using full document for extraction ({len(text)} characters)")
    
    # Enhanced prompt with TABLE-SPECIFIC instructions
    extraction_prompt = ChatPromptTemplate.from_template(
        """You are an expert document analyzer specializing in extracting structured data from tables and text.

DOCUMENT: {filename}
TYPE: {doc_type}

CONTENT:
{document}

CRITICAL INSTRUCTIONS - TABLES ARE PRIORITY:

1. IDENTIFY ALL TABLES (marked with TABLE START/END or containing "|" separators)
2. For KEY METRICS or STATISTICS tables:
   - Extract EVERY metric name and its value
   - State each metric in 3 different ways:
     a) Direct: "Metric Name: Value"
     b) Sentence: "The metric name is value"
     c) Question format: "Metric name equals value"

3. For LOCATION/BRANCH tables:
   - Count total entries
   - List each location with ALL details
   - Include addresses, phones, services

4. For any other tables:
   - Convert table data to natural language
   - Preserve all numerical values



FORMAT YOUR RESPONSE WITH THESE SECTIONS:

1. KEY METRICS & STATISTICS:
   (Extract from tables with metrics, KPIs, numbers)

2. BRANCH/LOCATION COUNT & LIST:
   (Count: X branches/offices)
   (List each with full details)

3. PRODUCTS/SERVICES:
   (Complete list with descriptions)

4. CONTACT INFORMATION:
   (All emails, phones, addresses)

5. COMPANY DETAILS:
   (Name, registration, industry)

6. IMPORTANT DATES & NUMBERS:
   (Any dates, financial figures, statistics)

Extract all information now, focusing on tables first:"""
    )
    
    try:
        llm = get_llm()
        
        logger.info("  Invoking LLM for TABLE-FOCUSED extraction...")
        response = llm.invoke(
            extraction_prompt.format(
                document=sample_text,
                filename=filename,
                doc_type=doc_metadata.get('document_type', 'unknown')
            )
        )
        
        extracted_content = response.content.strip()
        
        if len(extracted_content) < 50:
            logger.warning("  LLM extraction returned minimal content")
            return None
        
        logger.info(f"  ✓ LLM extraction successful: {len(extracted_content)} characters")
        logger.info(f"  Extraction preview:\n{extracted_content[:800]}...")
        
        # Create comprehensive summary
        summary = f"""╔══════════════════════════════════════════════════════════════╗
║  COMPREHENSIVE STRUCTURED SUMMARY - {filename}
╚══════════════════════════════════════════════════════════════╝

This summary contains all key information extracted from tables and text,
formatted to answer questions about metrics, statistics, counts, locations,
services, and other important details.

{extracted_content}

═══════════════════════════════════════════════════════════════
🔍 SEARCHABLE CONTENT:
This summary answers questions like:
- What are the key metrics for 2024?
- What are the company statistics?
- How many branches/offices/locations?
- What services are offered?
- What are the contact details?
- What is the company information?
- What are the financial figures?

═══════════════════════════════════════════════════════════════
📊 KEYWORDS: metrics, statistics, 2024, KPI, performance, key metrics,
   assets, customers, employees, satisfaction, rating, revenue, count,
   total, number, branch, location, office, service, product, contact,
   company, financial, data, information

═══════════════════════════════════════════════════════════════
"""
        
        return summary
        
    except Exception as e:
        logger.error(f"  ❌ LLM extraction failed: {e}", exc_info=True)
        return None

def create_smart_chunks(text: str, filename: str, file_id: str, doc_metadata: Dict) -> List[Dict]:
    """Create smart chunks with semantic awareness"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    text_chunks = text_splitter.split_text(text)
    
    chunk_dicts = []
    for i, chunk in enumerate(text_chunks):
        chunk_analysis = analyze_chunk_content(chunk)
        
        metadata = {
            "source": filename,
            "file_id": file_id,
            "chunk_id": i,
            "chunk_type": "content",
            "chunk_count": len(text_chunks),
            "chunk_size": len(chunk),
            "word_count": len(chunk.split()),
            "position": f"{i+1}/{len(text_chunks)}",
            "created_at": datetime.now().isoformat(),
            **doc_metadata,
            **chunk_analysis
        }
        
        chunk_with_context = add_chunk_context(chunk, i, len(text_chunks), filename)
        
        chunk_dicts.append({
            "text": chunk_with_context,
            "metadata": metadata
        })
    
    return chunk_dicts

def analyze_chunk_content(chunk: str) -> Dict:
    """Analyze chunk content for metadata"""
    
    chunk_lower = chunk.lower()
    
    entities = {
        "has_location_info": any(kw in chunk_lower for kw in 
            ["branch", "office", "location", "address", "city"]),
        "has_contact_info": any(kw in chunk_lower for kw in 
            ["phone", "email", "contact", "@"]),
        "has_pricing_info": any(kw in chunk_lower for kw in 
            ["price", "cost", "rate", "₹", "$"]),
        "has_metrics_info": any(kw in chunk_lower for kw in
            ["metric", "statistic", "kpi", "performance", "2024"]),
        "has_table_data": "TABLE" in chunk.upper() or "|" in chunk,
    }
    
    counts = {
        "branch_mentions": chunk_lower.count("branch") + chunk_lower.count("office"),
        "metric_mentions": chunk_lower.count("metric") + chunk_lower.count("statistic"),
    }
    
    return {**entities, **counts}

def add_chunk_context(chunk: str, chunk_index: int, total_chunks: int, filename: str) -> str:
    """Add contextual header to chunk"""
    
    context_header = f"[Document: {filename} | Part {chunk_index + 1}/{total_chunks}]\n\n"
    return context_header + chunk
