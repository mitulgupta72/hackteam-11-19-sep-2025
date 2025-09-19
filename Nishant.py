You said:
[9/19, 4:06 PM] Nishant Tcs: #!/usr/bin/env python3
"""
Simplified RAG Pipeline - One PDF → One JSON → Pinecone Cloud Only
"""

import json
import logging
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from mistralai import Mistral  # type: ignore
except Exception:
    Mistral = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

try:
    from pinecone import Pinecone  # type: ignore
except Exception:
    Pinecone = None

try:
    import nltk  # type: ignore
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from collections import Counter
    import re
except Exception:
    nltk = None
    from collections import Counter
    import re


class OCRResponseEncoder(json.JSONEncoder):
    """Custom JSON encoder for OCR response objects"""
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self.default(value)
                elif isinstance(value, list):
                    result[key] = [self.default(item) for item in value]
                else:
                    result[key] = value
            return result
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        else:
            return str(obj)


class TextChunker:
    """Splits text into overlapping chunks with table awareness."""

    def __init__(self, chunk_size: int = 2000, overlap: int = 300):
        self.chunk_size = chunk_size
        self.overlap = overlap
        if nltk:
            try:
                self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            except:
                self.tokenizer = None
        else:
            self.tokenizer = None

    def sentence_tokenize(self, text: str) -> List[str]:
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            # Simple fallback
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        if self._contains_table(text):
            return self._chunk_table_aware(text)

        sentences = self.sentence_tokenize(text)
        chunks: List[str] = []
        buffer: List[str] = []
        current_len = 0

        for sent in sentences:
            # If single sentence is too long, hard-split it
            if len(sent) > self.chunk_size:
                if buffer:
                    chunks.append(" ".join(buffer).strip())
                    buffer, current_len = [], 0
                for i in range(0, len(sent), self.chunk_size - self.overlap or self.chunk_size):
                    piece = sent[i:i + self.chunk_size]
                    chunks.append(piece.strip())
                continue

            if current_len + len(sent) + 1 <= self.chunk_size:
                buffer.append(sent)
                current_len += len(sent) + 1
            else:
                # Current sentence doesn't fit, so save buffer as a chunk
                if buffer:
                    chunks.append(" ".join(buffer).strip())
                    # Start new buffer with overlap from previous chunk
                    overlap_text = " ".join(buffer)[max(0, current_len - self.overlap):].strip()
                    buffer = self.sentence_tokenize(overlap_text)
                    current_len = len(overlap_text)
                else:
                    buffer = []
                    current_len = 0

                # Add current sentence to buffer
                buffer.append(sent)
                current_len += len(sent) + 1

        if buffer:
            chunks.append(" ".join(buffer).strip())

        # Final clean-up
        return [c for c in (ch.strip() for ch in chunks) if c]

    def _contains_table(self, text: str) -> bool:
        """Check if text contains table markers"""
        table_indicators = [
            "| :--: |",  # Markdown table separator
            "| :-- |",   # Markdown table separator
            "Material Issue Identified",  # Specific to our use case
            "| Sr. |",   # Table with Sr. column
            "| Category |",  # Table with Category column
        ]
        return any(indicator in text for indicator in table_indicators)

    def _chunk_table_aware(self, text: str) -> List[str]:
        """Chunk text while preserving table integrity"""
        # Use sentence-based chunking with increased size/overlap for tables
        sentences = self.sentence_tokenize(text)
        chunks: List[str] = []
        buffer: List[str] = []
        current_len = 0

        for sent in sentences:
            if len(sent) > self.chunk_size:
                if buffer:
                    chunks.append(" ".join(buffer).strip())
                    buffer, current_len = [], 0
                for i in range(0, len(sent), self.chunk_size - self.overlap or self.chunk_size):
                    piece = sent[i:i + self.chunk_size]
                    chunks.append(piece.strip())
                continue

            if current_len + len(sent) + 1 <= self.chunk_size:
                buffer.append(sent)
                current_len += len(sent) + 1
            else:
                if buffer:
                    chunks.append(" ".join(buffer).strip())
                    overlap_text = " ".join(buffer)[max(0, current_len - self.overlap):].strip()
                    buffer = self.sentence_tokenize(overlap_text)
                    current_len = len(overlap_text)
                else:
                    buffer = []
                    current_len = 0

                buffer.append(sent)
                current_len += len(sent) + 1

        if buffer:
            chunks.append(" ".join(buffer).strip())

        return [c for c in (ch.strip() for ch in chunks) if c]


class KeywordExtractor:
    """Extract keywords from text using TF-IDF and RAKE-style extraction."""

    def __init__(self):
        if nltk:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.stemmer = PorterStemmer()
            except:
                self.stop_words = set()
                self.stemmer = None
        else:
            self.stop_words = set()
            self.stemmer = None

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        # Remove special characters and split
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) > 2 and w not in self.stop_words]

    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords using TF-IDF approach"""
        if not text:
            return []
        
        # Tokenize
        words = self.tokenize(text)
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get most common words
        common_words = word_counts.most_common(max_keywords)
        
        # Extract keywords
        keywords = []
        for word, count in common_words:
            if count > 1:  # Only words that appear more than once
                keywords.append(word)
        
        # Add some key phrases (simple approach)
        phrases = self._extract_phrases(text)
        keywords.extend(phrases[:5])  # Add top 5 phrases
        
        return keywords[:max_keywords]

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract key phrases using simple n-gram approach"""
        words = self.tokenize(text)
        phrases = []
        
        # Bigrams
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 5:  # Only meaningful phrases
                phrases.append(phrase)
        
        # Trigrams
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase) > 10:  # Only meaningful phrases
                phrases.append(phrase)
        
        return phrases


class EmbeddingGenerator:
    """Generate OpenAI embeddings for chunks."""

    def __init__(self, model: str = "text-embedding-3-large"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is required")
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed")
        
        self.client = OpenAI()
        self.model = model

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"🤖 Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.info(f"✅ Generated {len(batch_embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings


class PineconeUploader:
    """Upload embeddings to Pinecone."""

    def __init__(self, index_name: str, namespace: Optional[str] = None):
        if not os.getenv("PINECONE_API_KEY"):
            raise RuntimeError("PINECONE_API_KEY environment variable is required")
        if Pinecone is None:
            raise RuntimeError("Pinecone SDK not installed")
        
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

    def upload_embeddings(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Upload chunks with embeddings to Pinecone."""
        logger.info(f"🌲 Uploading {len(chunks)} chunks to Pinecone...")
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding:  # Skip if embedding generation failed
                continue
                
            vector = {
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "keywords": chunk["keywords"],
                    "source": chunk["source"],
                    "chunk_index": i
                }
            }
            vectors.append(vector)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            logger.info(f"📤 Uploading batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
                logger.info(f"✅ Uploaded {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Failed to upload batch: {e}")
                return False
        
        logger.info(f"🎉 Successfully uploaded {len(vectors)} vectors to Pinecone")
        return True


def process_single_pdf(pdf_path: str, output_dir: str = ".") -> Optional[str]:
    """Process a single PDF: OCR → JSON → Chunking → Keywords → Embeddings → Pinecone."""
    logger.info(f"📄 Processing PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    
    # Initialize components
    if Mistral is None:
        logger.error("Mistral SDK not installed")
        return None
    
    mistral = Mistral(api_key=os.getenv("MISTRALAI_API_KEY"))
    chunker = TextChunker(chunk_size=2000, overlap=300)
    keyword_extractor = KeywordExtractor()
    embedding_generator = EmbeddingGenerator()
    
    try:
        # Step 1: OCR
        logger.info("🔍 Running OCR...")
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        # Encode PDF to base64
        base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        
        ocr_response = mistral.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            }
        )
        
        # Step 2: Save OCR response as JSON
        pdf_name = Path(pdf_path).stem
        ocr_json_path = os.path.join(output_dir, f"{pdf_name}_ocr.json")
        
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_response, f, cls=OCRResponseEncoder, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 OCR response saved to: {ocr_json_path}")
        
        # Step 3: Extract text and chunk
        logger.info("✂️ Extracting text and chunking...")
        text = ocr_response.text if hasattr(ocr_response, 'text') else str(ocr_response)
        chunks = chunker.chunk(text)
        logger.info(f"📝 Created {len(chunks)} chunks")
        
        # Step 4: Extract keywords for each chunk
        logger.info("🔑 Extracting keywords...")
        chunk_data = []
        for i, chunk_text in enumerate(chunks):
            keywords = keyword_extractor.extract_keywords(chunk_text)
            chunk_data.append({
                "id": f"{pdf_name}::chunk::{i+1}",
                "text": chunk_text,
                "keywords": keywords,
                "source": f"{pdf_name}_ocr.json"
            })
        
        # Step 5: Generate embeddings
        logger.info("🤖 Generating embeddings...")
        texts = [chunk["text"] for chunk in chunk_data]
        embeddings = embedding_generator.generate_embeddings_batch(texts)
        
        # Step 6: Upload to Pinecone
        logger.info("🌲 Uploading to Pinecone...")
        pinecone_uploader = PineconeUploader("leagalaitanish")
        success = pinecone_uploader.upload_embeddings(chunk_data, embeddings)
        
        if success:
            logger.info(f"✅ Successfully processed {pdf_path}")
            return ocr_json_path
        else:
            logger.error(f"❌ Failed to upload to Pinecone for {pdf_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}")
        return None


def main():
    """Main function to process PDFs."""
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables first
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Process PDFs: OCR → JSON → Pinecone")
    parser.add_argument("pdf_path", help="Path to PDF file or directory containing PDFs")
    parser.add_argument("--output-dir", default=".", help="Output directory for JSON files")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    # Check required environment variables
    required_vars = ["MISTRALAI_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return
    
    pdf_path = Path(args.pdf_path)
    
    if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
        # Process single PDF
        logger.info("🚀 Starting single PDF processing...")
        result = process_single_pdf(str(pdf_path), args.output_dir)
        if result:
            logger.info(f"🎉 Processing completed! OCR JSON saved to: {result}")
        else:
            logger.error("❌ Processing failed!")
    
    elif pdf_path.is_dir():
        # Process directory of PDFs
        logger.info("🚀 Starting batch PDF processing...")
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.error("No PDF files found in directory")
            return
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files")
        
        # Process PDFs concurrently
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_pdf = {
                executor.submit(process_single_pdf, str(pdf_file), args.output_dir): pdf_file 
                for pdf_file in pdf_files
            }
            
            successful = 0
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                        logger.info(f"✅ {pdf_file.name} → {result}")
                    else:
                        logger.error(f"❌ Failed to process {pdf_file.name}")
                except Exception as e:
                    logger.error(f"❌ Error processing {pdf_file.name}: {e}")
        
        logger.info(f"🎉 Batch processing completed! {successful}/{len(pdf_files)} PDFs processed successfully")
    
    else:
        logger.error("Invalid path. Please provide a PDF file or directory containing PDFs.")


if __name__ == "__main__":
    main()
[9/19, 4:06 PM] Nishant Tcs: pdf retrival and pinecone push
[9/19, 4:06 PM] Nishant Tcs: #!/usr/bin/env python3
"""
Simplified RAG Retrieval Pipeline - Pinecone Only
One PDF → One JSON → Pinecone Cloud Only
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

try:
    from pinecone import Pinecone  # type: ignore
except Exception:
    Pinecone = None


class SimpleRetriever:
    """Simplified retriever using only Pinecone cloud."""

    def __init__(
        self,
        pinecone_index: str,
        pinecone_namespace: Optional[str] = None,
        openai_model: str = "text-embedding-3-large",
    ) -> None:
        logger.info("🔧 Initializing SimpleRetriever...")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is required")
        if not os.getenv("PINECONE_API_KEY"):
            raise RuntimeError("PINECONE_API_KEY environment variable is required")
        
        self.pinecone_index = pinecone_index
        self.pinecone_namespace = pinecone_namespace
        self.openai_model = openai_model
        
        # Lazy init clients
        self._openai_client = None
        self._pinecone_index_client = None
        
        logger.info("✅ SimpleRetriever initialized successfully")

    def _openai(self) -> OpenAI:
        if self._openai_client is None:
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not installed")
            self._openai_client = OpenAI()
        return self._openai_client

    def _pinecone(self) -> Any:
        if self._pinecone_index_client is None:
            if Pinecone is None:
                raise RuntimeError("Pinecone SDK not installed")
            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            self._pinecone_index_client = pc.Index(self.pinecone_index)
        return self._pinecone_index_client

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query using OpenAI."""
        logger.info(f"🤖 Generating embedding using model: {self.openai_model}")
        client = self._openai()
        resp = client.embeddings.create(model=self.openai_model, input=[query])
        embedding = resp.data[0].embedding
        logger.info(f"✅ Embedding generated: {len(embedding)} dimensions")
        return embedding

    def _expand_query_for_material_issues(self, query: str) -> List[str]:
        """Expand query with additional terms to better find material issues chunks."""
        query_lower = query.lower()
        
        # Check if query is about counting or listing material issues
        material_issue_indicators = [
            "material issues", "material issue", "how many material",
            "count material", "list material", "identify material",
            "responsible business conduct issues", "sustainability issues"
        ]
        
        if any(indicator in query_lower for indicator in material_issue_indicators):
            # Add specific material issue terms that might be in separate chunks
            expanded_terms = [
                "Green Buildings", "Climate Change", "Business Ethics",
                "Customer Data Privacy", "Food Quality", "Waste Management",
                "Customer Satisfaction", "Resilient Business Strategy",
                "Energy and Emissions Management", "Water and Effluent management"
            ]
            return [query] + expanded_terms
        
        return [query]

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Search using Pinecone with intelligent query expansion
        
        This is the core retrieval method that:
        1. Expands queries for better material issues retrieval
        2. Generates embeddings for multiple query variations
        3. Searches Pinecone with weighted scoring
        4. Returns ranked results for LLM processing
        """
        logger.info(f"🔍 Starting Pinecone search for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        logger.info(f"📊 Search parameters: top_k={top_k}")
        
        # ========================================
        # STEP 1: INTELLIGENT QUERY EXPANSION
        # ========================================
        # LLM Layer 0: Query preprocessing and expansion
        # This automatically expands queries about "material issues" 
        # with specific terms like "Green Buildings", "Climate Change", etc.
        # This ensures we don't miss relevant chunks that don't contain
        # the exact query terms but are semantically related
        expanded_queries = self._expand_query_for_material_issues(query)
        logger.info(f"🔍 Query expansion: {len(expanded_queries)} search terms generated")
        
        # ========================================
        # STEP 2: EMBEDDING GENERATION
        # ========================================
        # Generate OpenAI embeddings for all expanded queries
        # Each query variation gets its own embedding vector
        logger.info("🤖 Generating query embeddings using OpenAI...")
        query_embeddings = []
        for expanded_query in expanded_queries:
            qvec = self._embed_query(expanded_query)
            if qvec:
                query_embeddings.append((expanded_query, qvec))
        
        if not query_embeddings:
            return {"success": False, "message": "Failed to generate query embeddings", "data": {"results": []}}
        
        logger.info(f"✅ Query embeddings generated: {len(query_embeddings)} embeddings")

        # ========================================
        # STEP 3: VECTOR SEARCH (Pinecone)
        # ========================================
        # Search Pinecone with multiple query variations
        # Each query gets weighted differently (original query gets higher weight)
        # This ensures comprehensive retrieval of relevant chunks
        logger.info("🔍 Performing Pinecone search...")
        all_results = {}
        for i, (expanded_query, qvec) in enumerate(query_embeddings):
            logger.info(f"🌲 Querying Pinecone index: top_k={top_k}, namespace={self.pinecone_namespace}")
            res = self._pinecone().query(
                vector=qvec,
                top_k=top_k,
                include_metadata=True,
                namespace=self.pinecone_namespace,
            )
            
            # Weight the first query (original) higher than expanded queries
            # This ensures original query intent is prioritized
            weight = 1.0 if i == 0 else 0.7
            for match in res.get("matches", []):
                chunk_id = match["id"]
                score = float(match.get("score", 0.0)) * weight
                metadata = match.get("metadata", {})
                
                if chunk_id not in all_results or score > all_results[chunk_id]["score"]:
                    all_results[chunk_id] = {
                        "id": chunk_id,
                        "score": score,
                        "text": metadata.get("text", ""),
                        "keywords": metadata.get("keywords", []),
                        "source": metadata.get("source", "unknown")
                    }
        
        logger.info(f"✅ Pinecone search completed: {len(all_results)} results")

        # ========================================
        # STEP 4: RESULT RANKING & RETURN
        # ========================================
        # Sort results by combined scores and return top chunks
        # These chunks will be passed to LLM for answer synthesis
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        
        logger.info(f"🎯 Search completed successfully: {len(sorted_results)} chunks retrieved")
        
        return {
            "success": True,
            "message": f"Retrieved {len(sorted_results)} chunks",
            "data": {
                "query": query,
                "total_chunks": len(sorted_results),
                "results": sorted_results
            }
        }


class LLMAnswerer:
    """LLM-based answer synthesis over retrieved chunks."""

    def __init__(self, model: str = "gpt-4.1") -> None:
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed.")
        self.model = model
        self.client = OpenAI()

    def _chat(self, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    
    def _chat_stream(self, messages: List[Dict[str, str]], temperature: float = 1.0):
        """Stream chat completion for real-time response"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        return stream

    def assess_and_refine(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        LLM Layer 1: Query Assessment & Refinement
        
        This LLM layer evaluates whether the retrieved chunks contain sufficient
        information to answer the query. If not, it can suggest a refined query.
        
        This is an optional layer that can be used for query optimization.
        """
        logger.info(f"🧠 LLM Assessment: Evaluating {len(chunks)} chunks for query relevance...")
        
        # Prepare context for LLM assessment
        context = []
        for i, c in enumerate(chunks[:8]):  # Limit context size
            context.append(f"[Chunk {i+1}]\n{c.get('text','')}")
        ctx = "\n\n".join(context)
        
        # LLM Layer 1: Assessment prompt
        system = (
           """ Generate concise, clear, and relevant answers based strictly on provided information. Avoid fluff and repetition. Highlight only key facts and essential details.and include the full information about the question provided as much as possible. Keep language simple and professional."""
        )
        user = (
            f"Query: {query}\n\nChunks:\n{ctx}\n\n"
            "Rate feasibility (true/false) and suggest a refined query if needed. "
            "Respond in JSON: {\"feasible\": boolean, \"refined_query\": \"string\", \"reason\": \"string\"}"
        )
        
        logger.info("🤖 Sending assessment request to LLM...")
        response = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=1.0)
        
        logger.info(f"✅ LLM assessment response received: {response[:200]}...")
        
        try:
            result = json.loads(response)
            logger.info(f"✅ LLM assessment parsed: feasible={result.get('feasible')}, refined_query={bool(result.get('refined_query'))}")
            return result
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse LLM assessment: {e}")
            return {"feasible": True, "refined_query": None, "reason": "Assessment failed"}

    def synthesize(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        LLM Layer 2: Answer Synthesis (Main LLM Processing)
        
        This is the core LLM layer that:
        1. Takes all retrieved chunks as context
        2. Uses GPT-5 to generate comprehensive answers
        3. Follows specific prompts for legal document analysis
        4. Ensures all relevant information is included
        5. Returns a coherent, accurate answer
        
        This is where the "magic" happens - the LLM reads through all chunks
        and synthesizes a comprehensive answer that addresses the user's query.
        """
        logger.info(f"🎯 LLM Synthesis: Generating answer from {len(chunks)} chunks...")
        
        # Prepare context from all retrieved chunks
        context = []
        for i, c in enumerate(chunks[:12]):
            context.append(f"[Chunk {i+1}]\n{c.get('text','')}")
        ctx = "\n\n".join(context)
        
        logger.info(f"📝 Context prepared: {len(ctx)} characters from {min(len(chunks), 12)} chunks")
        
        # LLM Layer 2: Answer synthesis prompt
        # This prompt is specifically designed for legal document analysis
        system = (
            "You are a legal RAG answerer. Extract ALL relevant information from the provided chunks.\n"
            "Be thorough and comprehensive. Look for tables, lists, and detailed information.\n"
            "If asking for counts/numbers, scan through ALL chunks to find complete information.\n"
            "Answer precisely, cite facts only from the provided chunks."
        )
        user = (
            f"Query: {query}\n\nContext Chunks:\n{ctx}\n\n"
            "Extract ALL relevant information. Be comprehensive and thorough. "
            "If the query asks for a count or list, make sure to find ALL items mentioned across the chunks."
        )
        
        logger.info("🤖 Sending synthesis request to LLM...")
        response = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=1.0)
        
        logger.info(f"✅ LLM synthesis completed: {len(response)} characters generated")
        return response
    
    def synthesize_stream(self, query: str, chunks: List[Dict[str, Any]]):
        """
        Stream LLM synthesis for real-time response display
        """
        logger.info(f"🎯 LLM Synthesis: Generating streaming answer from {len(chunks)} chunks...")
        
        # Prepare context from limited chunks for speed (max 6 chunks, 2000 chars each)
        context = []
        total_chars = 0
        max_chars = 12000  # Limit total context for faster processing
        
        for i, c in enumerate(chunks[:6]):  # Limit to 6 chunks max
            chunk_text = c.get('text', '')[:2000]  # Limit each chunk to 2000 chars
            if total_chars + len(chunk_text) > max_chars:
                break
            context.append(f"[Chunk {i+1}]\n{chunk_text}")
            total_chars += len(chunk_text)
        
        ctx = "\n\n".join(context)
        
        logger.info(f"📝 Context prepared: {len(ctx)} characters from {len(context)} chunks")
        
        # LLM Layer 2: Answer synthesis prompt - OPTIMIZED
        # Shorter, more focused prompt for faster generation
        system = (
            "You are a legal document assistant. Extract relevant information from the provided chunks.\n"
            "Be concise but comprehensive. Look for key facts, numbers, and important details.\n"
            "Answer precisely based only on the provided chunks."
        )
        user = (
            f"Query: {query}\n\nContext:\n{ctx}\n\n"
            "Provide a clear, concise answer based on the context above."
        )
        
        logger.info("🤖 Sending streaming synthesis request to LLM...")
        return self._chat_stream([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=1.0)


def main():
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run simplified Pinecone-only retrieval.")
    parser.add_argument("--query", type=str, default=None, help="Query text. If omitted, will prompt.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top chunks to retrieve.")
    parser.add_argument("--pinecone-index", type=str, default="leagalaitanish", help="Pinecone index name.")
    parser.add_argument("--pinecone-namespace", type=str, default=None, help="Pinecone namespace.")
    parser.add_argument("--openai-model", type=str, default="text-embedding-3-large", help="OpenAI embedding model.")
    parser.add_argument("--answer", action="store_true", help="Run LLM answer synthesis.")
    parser.add_argument("--max-refine-steps", type=int, default=1, help="Max query refinement iterations.")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1", help="OpenAI chat model for synthesis.")
    
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Enter your query: ")

    try:
        logger.info("🚀 Starting Simplified RAG Retrieval Pipeline")
        logger.info(f"⚙️ Configuration: top_k={args.top_k}, pinecone_index={args.pinecone_index}, answer_mode={args.answer}")
        
        retriever = SimpleRetriever(
            pinecone_index=args.pinecone_index,
            pinecone_namespace=args.pinecone_namespace,
            openai_model=args.openai_model,
        )
        
        res = retriever.search(query, top_k=args.top_k)

        if not args.answer:
            print(json.dumps({
                "success": True,
                "message": f"Retrieved {len(res['data']['results'])} chunks",
                "data": {
                    "query": query,
                    "total_chunks": len(res['data']['results']),
                    "chunks": res['data']['results']
                }
            }, indent=2, ensure_ascii=False))
            return

        # LLM-based answer synthesis
        logger.info("🧠 Starting LLM-based processing...")
        validator = LLMAnswerer(model=args.llm_model)
        current_query = query
        results = res
        
        for step in range(max(0, args.max_refine_steps) + 1):
            logger.info(f"🔄 Refinement step {step + 1}/{args.max_refine_steps + 1}")
            chunks = results["data"]["results"]
            assessment = validator.assess_and_refine(current_query, chunks)
            
            if assessment.get("feasible") or not assessment.get("refined_query"):
                logger.info(f"✅ Assessment complete: feasible={assessment.get('feasible')}, reason={assessment.get('reason', 'N/A')}")
                break
            
            # Re-run retrieval with refined query
            current_query = assessment.get("refined_query")
            logger.info(f"🔄 Refining query: '{current_query}'")
            results = retriever.search(current_query, top_k=args.top_k)

        # Synthesize final answer
        logger.info("🎯 Starting final answer synthesis...")
        final_chunks = results["data"]["results"]
        answer = validator.synthesize(current_query, final_chunks)
        
        print(json.dumps({
            "success": True,
            "message": "answer synthesized",
            "data": {
                "original_query": query,
                "final_query": current_query,
                "answer": answer
            }
        }, indent=2, ensure_ascii=False))
        
        logger.info("🎉 RAG Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        print(json.dumps({"success": False, "message": f"Retrieval failed: {e}", "data": None}, indent=2))


if __name__ == "__main__":
    main()
[9/19, 4:09 PM] Nishant Tcs: Is me s mistral hata k normal pdf extractor laga don