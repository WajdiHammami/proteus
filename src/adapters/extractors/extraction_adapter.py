

"""
SCHEMA-DRIVEN EXTRACTION TECHNIQUES CHEAT SHEET
=============================================

1. GENERATIVE LLM TECHNIQUES (Flexible, Semantic)
   ----------------------------------------------
   - Native Constrained Decoding (Grammar-Based): 
     Forces LLM token generation at the logit level to strictly match a 
     Pydantic/JSON schema (e.g., OpenAI Structured Outputs, Outlines). 
     -> 100% reliable syntax, zero parsing errors.
   - Function / Tool Calling: 
     Maps YAML schema fields to tool parameter definitions. Supported across 
     most providers (OpenAI, Ollama). Good balance of flexibility and structure.
   - Prompt Engineering + Pydantic Repair: 
     Prompt model for JSON -> validate with Pydantic -> feed validation errors 
     back into a reflection loop if it fails.

2. SPECIALIZED SMALL AI MODELS (Fast, Cost-Effective)
   -------------------------------------------------
   - Information Extraction SLMs (e.g., NuExtract, fine-tuned Llama-3-8B): 
     Compact models fine-tuned specifically for structured field extraction. 
     -> Extremely fast, runnable locally, very low per-doc cost.
   - Multimodal Vision-Language Models (VLMs like GPT-4o-vision, LayoutLM): 
     Processes visual layouts, tables, and multi-column documents directly 
     as images without losing structural hierarchy.

3. DETERMINISTIC & HEURISTIC TECHNIQUES (Instant, $0 Cost)
   ------------------------------------------------------
   - DOM / XPath / CSS Selectors (BeautifulSoup, lxml): 
     Direct element tree extraction. Instant, but brittle (breaks when site markup changes).
   - Regex & Pattern Matching / NER (spaCy): 
     Pulls explicit values (dates, currency, IDs) via pattern recognition.
   - Hybrid / Cascading Architecture (Recommended): 
     Chain them -> Try fast Regex/XPath first -> Fallback to an SLM for missing fields 
     -> Use a constrained top-tier LLM only for complex ambiguity.
"""