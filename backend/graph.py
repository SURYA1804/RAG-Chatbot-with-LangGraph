from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import logging
from dotenv import load_dotenv

from vectore_store import VectorStoreManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Domain-specific synonyms
DOMAIN_SYNONYMS = {
    "branch": ["office", "location", "center", "outlet", "agency"],
    "loan": ["credit", "financing", "lending", "advance"],
    "eligibility": ["qualification", "criteria", "requirements", "conditions"],
    "apply": ["application", "request", "submit", "register"],
    "interest rate": ["rate", "APR", "interest", "percentage"],
    "cost": ["price", "fee", "charge", "amount"],
}

class GraphState(TypedDict):
    """State definition for the RAG agent"""
    messages: List[dict]
    query: str
    query_variations: List[str]
    intent_info: str
    documents: List[str]
    sources: List[str]
    generation: str
    documents_relevant: bool

def create_rag_agent(vector_store : VectorStoreManager):
    """Create a production-ready RAG agent with natural language handling"""
    
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_retries=3
    )
    
    def add_conversational_context(state: GraphState) -> GraphState:
        """Add context from conversation history for follow-up questions"""
        messages = state["messages"]
        current_query = messages[-1]["content"]
        
        logger.info("="*80)
        logger.info("STEP 1: CONVERSATIONAL CONTEXT")
        logger.info(f"Original query: {current_query}")
        
        # Check if it's a follow-up question
        if len(messages) > 2:  # Has previous conversation
            recent_messages = messages[-6:] if len(messages) > 6 else messages[:-1]
            
            context_prompt = ChatPromptTemplate.from_template(
                """Given the conversation history, rewrite the current question to be standalone with full context.

Conversation History:
{history}

Current Question: {question}

Rewrite as standalone question with context:"""
            )
            
            history = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_messages
            ])
            
            try:
                response = llm.invoke(
                    context_prompt.format(history=history, question=current_query)
                )
                
                contextualized_query = response.content.strip()
                logger.info(f"Contextualized query: {contextualized_query}")
                
                return {
                    **state,
                    "query": contextualized_query
                }
            except Exception as e:
                logger.error(f"Error contextualizing: {e}")
        
        logger.info("No context needed - first query or standalone")
        return {
            **state,
            "query": current_query
        }
    
    def classify_intent(state: GraphState) -> GraphState:
        """Classify user intent to guide retrieval"""
        query = state["query"]
        
        logger.info("\nSTEP 2: INTENT CLASSIFICATION")
        
        intent_prompt = ChatPromptTemplate.from_template(
            """Classify the user's intent and extract key information.

Question: {question}

Classify into ONE primary category:
- LOCATION_INFO: Questions about branches, addresses, offices, where to visit
- PRODUCT_INFO: Questions about loans, services, products, offerings
- ELIGIBILITY: Questions about who qualifies, requirements, criteria
- PROCESS: Questions about how to do something, application steps
- PRICING: Questions about rates, costs, fees, charges
- CONTACT: Questions about phone, email, customer support
- COMPANY_INFO: General company information, history, overview
- FAQ: General questions, help

Also extract:
- Key entities mentioned (city names, product names, numbers)
- What type of answer is needed (count, list, specific detail, explanation)

Format your response EXACTLY as:
INTENT: [category]
ENTITIES: [comma-separated list]
ANSWER_TYPE: [count/list/specific/explanation]

Respond:"""
        )
        
        try:
            response = llm.invoke(intent_prompt.format(question=query))
            intent_info = response.content.strip()
            
            logger.info(f"Intent classification:\n{intent_info}")
            
            return {
                **state,
                "intent_info": intent_info
            }
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {
                **state,
                "intent_info": "INTENT: GENERAL\nENTITIES: none\nANSWER_TYPE: explanation"
            }
    
    def reformulate_query(state: GraphState) -> GraphState:
        """Reformulate query into multiple variations"""
        query = state["query"]
        intent_info = state.get("intent_info", "")
        
        logger.info("\nSTEP 3: QUERY REFORMULATION")
        
        # Extract intent
        intent = "GENERAL"
        if "INTENT:" in intent_info:
            intent = intent_info.split("INTENT:")[1].split("\n")[0].strip()
        
        reformulation_prompt = ChatPromptTemplate.from_template(
            """You are a query reformulation expert. Generate 4 alternative phrasings of the question.

Original Question: {question}
User Intent: {intent}

Generate variations that:
1. Use synonyms (office→branch, cost→price, get→apply for)
2. Rephrase structure (How many X? → Total number of X, List all X)
3. Be more specific (add relevant domain terms)
4. Use different question formats

Provide exactly 4 variations, one per line, without numbering:"""
        )
        
        try:
            response = llm.invoke(
                reformulation_prompt.format(question=query, intent=intent)
            )
            
            # Parse variations
            variations = [v.strip().lstrip('0123456789.- ') for v in response.content.split('\n') 
                         if v.strip() and len(v.strip()) > 10]
            
            # Include original query first
            all_queries = [query] + variations[:4]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in all_queries:
                q_lower = q.lower()
                if q_lower not in seen:
                    seen.add(q_lower)
                    unique_queries.append(q)
            
            logger.info(f"Generated {len(unique_queries)} query variations:")
            for i, q in enumerate(unique_queries, 1):
                logger.info(f"  {i}. {q}")
            
            return {
                **state,
                "query_variations": unique_queries
            }
        except Exception as e:
            logger.error(f"Error reformulating: {e}")
            return {
                **state,
                "query_variations": [query]
            }
    
    def retrieve_with_intent(state: GraphState) -> GraphState:
        """Retrieve documents using intent and query variations"""
        query = state["query"]
        variations = state.get("query_variations", [query])
        intent_info = state.get("intent_info", "")
        
        logger.info("\nSTEP 4: INTELLIGENT RETRIEVAL")
        logger.info("="*80)
        
        # Extract intent
        intent = "GENERAL"
        if "INTENT:" in intent_info:
            intent = intent_info.split("INTENT:")[1].split("\n")[0].strip()
        
        logger.info(f"Retrieving for intent: {intent}")
        
        all_results = []
        seen_texts = set()
        
        # Search with each variation
        for i, var_query in enumerate(variations, 1):
            logger.info(f"\nSearching variation {i}/{len(variations)}: {var_query}")
            
            # Higher k for better coverage
            k_value = 15 if i == 1 else 10  # More results for original query
            results = vector_store.search(var_query, k=k_value)
            
            logger.info(f"  Retrieved {len(results)} documents")
            
            for doc in results:
                text = doc["text"]
                if text not in seen_texts:
                    # Calculate relevance boost based on intent
                    doc["relevance_boost"] = calculate_relevance_boost(
                        doc, intent, text
                    )
                    
                    all_results.append(doc)
                    seen_texts.add(text)
        
        # Sort by adjusted distance (lower is better)
        all_results.sort(key=lambda x: x.get("distance", 1.0) - x.get("relevance_boost", 0))
        
        logger.info(f"\nRETRIEVAL STATS:")
        logger.info(f"Total unique documents: {len(all_results)}")
        
        # Log top 5 results
        logger.info("\nTop 5 documents:")
        for i, doc in enumerate(all_results[:5], 1):
            logger.info(f"\n  {i}. Source: {doc['metadata'].get('source', 'Unknown')}")
            logger.info(f"     Chunk ID: {doc['metadata'].get('chunk_id', 'N/A')}")
            logger.info(f"     Distance: {doc.get('distance', 'N/A'):.4f}")
            logger.info(f"     Boost: {doc.get('relevance_boost', 0):.2f}")
            logger.info(f"     Preview: {doc['text'][:150]}...")
        
        # Check for structured summaries
        structured_count = sum(1 for doc in all_results 
                              if doc['metadata'].get('type') == 'structured_summary')
        logger.info(f"\nStructured summary chunks: {structured_count}")
        logger.info(f"Regular content chunks: {len(all_results) - structured_count}")
        logger.info("="*80)
        
        documents = [doc["text"] for doc in all_results[:30]]
        sources = [doc["metadata"].get("source", "Unknown") for doc in all_results[:30]]
        
        return {
            **state,
            "documents": documents,
            "sources": sources,
            "documents_relevant": False
        }
    
    def calculate_relevance_boost(doc: dict, intent: str, text: str) -> float:
        """Calculate relevance boost based on intent matching"""
        boost = 0.0
        text_lower = text.lower()
        
        intent_keywords = {
            "LOCATION_INFO": ["branch", "office", "location", "address", "visit", "center"],
            "PRODUCT_INFO": ["loan", "service", "product", "offer", "finance"],
            "ELIGIBILITY": ["eligibility", "qualify", "requirement", "criteria", "condition"],
            "PROCESS": ["application", "apply", "process", "step", "how to", "procedure"],
            "PRICING": ["rate", "price", "cost", "fee", "charge", "interest", "apr"],
            "CONTACT": ["phone", "email", "contact", "support", "helpline"],
            "COMPANY_INFO": ["company", "about", "overview", "history", "mission"],
        }
        
        keywords = intent_keywords.get(intent, [])
        
        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in text_lower)
        
        # Boost based on matches (0.02 per match, max 0.15)
        if matches > 0:
            boost = min(matches * 0.03, 0.15)
        
        # Extra boost for structured summaries
        if doc['metadata'].get('type') == 'structured_summary':
            boost += 0.1
        
        return boost
    
    def check_relevance(state: GraphState) -> GraphState:
        """Check if retrieved documents are relevant - LENIENT VERSION"""
        query = state["query"]
        documents = state["documents"]
        
        logger.info("\n" + "="*80)
        logger.info("STEP 5: RELEVANCE CHECK")
        logger.info("="*80)
        
        if not documents or len(documents) == 0:
            logger.warning("No documents to check relevance")
            return {
                **state,
                "documents_relevant": False
            }
        
        # Enhanced relevance prompt with explicit examples
        relevance_prompt = ChatPromptTemplate.from_template(
            """You are a relevance checker. Determine if the documents can help answer the question.

    Question: {question}

    Documents (showing 3 samples):
    {documents}

    IMPORTANT RULES - BE LENIENT:

    1. If documents contain CONTACT INFORMATION (emails, phones) and user asks for "contact", "email", "phone", "support" → Answer YES

    2. If documents contain COMPANY INFORMATION and user asks about the company → Answer YES

    3. If documents contain LOCATION/BRANCH data and user asks about locations → Answer YES

    4. If documents contain METRICS/STATISTICS and user asks about metrics/numbers → Answer YES

    5. If documents contain ANY information that could partially answer the question → Answer YES

    6. Only answer NO if documents are COMPLETELY UNRELATED to the topic
    (e.g., user asks about cars but docs are about banking)

    EXAMPLES:
    - Question: "What is the customer support email?"
    Documents contain: "suryafinance@customersupport.com"
    Answer: YES

    - Question: "How many branches?"
    Documents contain: Branch locations
    Answer: YES

    - Question: "What is the capital of France?"
    Documents are about finance company
    Answer: NO (completely unrelated)

    Based on these rules, can the documents help answer the question?

    Answer with ONE WORD ONLY (YES or NO):"""
        )
        
        # Show more content from each document (increased from 600 to 1000 chars)
        docs_sample = "\n\n---\n\n".join([
            f"Document {i+1} (from {doc[:80]}...):\n{doc[:1000]}" 
            for i, doc in enumerate(documents[:3])
        ])
        
        try:
            logger.info(f"Checking relevance for query: {query[:100]}...")
            logger.info(f"Document count: {len(documents)}")
            
            response = llm.invoke(
                relevance_prompt.format(question=query, documents=docs_sample)
            )
            
            response_text = response.content.strip().lower()
            is_relevant = "yes" in response_text
            
            logger.info(f"LLM relevance response: {response.content.strip()}")
            
            # CRITICAL FAILSAFE: Override incorrect NO responses
            if not is_relevant and len(documents) > 0:
                # Check if query is asking for info that documents likely contain
                query_lower = query.lower()
                
                # Keywords that suggest the documents ARE relevant
                contact_keywords = ["email", "contact", "phone", "support", "call", "reach"]
                location_keywords = ["branch", "office", "location", "address", "where"]
                info_keywords = ["information", "details", "tell me", "what is", "how many"]
                
                asking_for_contact = any(kw in query_lower for kw in contact_keywords)
                asking_for_location = any(kw in query_lower for kw in location_keywords)
                asking_for_info = any(kw in query_lower for kw in info_keywords)
                
                # Check if any document contains relevant info
                docs_text = " ".join(documents[:3]).lower()
                has_email = "@" in docs_text or "email" in docs_text
                has_phone = "phone" in docs_text or any(char.isdigit() for char in docs_text[:500])
                has_branches = "branch" in docs_text or "office" in docs_text or "location" in docs_text
                
                # Override if there's a clear match
                if (asking_for_contact and (has_email or has_phone)):
                    logger.warning("⚠️ OVERRIDE: Query asks for contact, documents have contact info → Forcing YES")
                    is_relevant = True
                elif (asking_for_location and has_branches):
                    logger.warning("⚠️ OVERRIDE: Query asks for locations, documents have location info → Forcing YES")
                    is_relevant = True
                elif asking_for_info and len(documents) > 0:
                    logger.warning("⚠️ OVERRIDE: Query asks for information, documents exist → Forcing YES")
                    is_relevant = True
                else:
                    logger.warning(f"⚠️ Relevance check returned NO, but could not auto-override")
            
            relevance_status = "✓ RELEVANT" if is_relevant else "✗ NOT RELEVANT"
            logger.info(f"Final decision: {relevance_status}")
            logger.info("="*80)
            
            return {
                **state,
                "documents_relevant": is_relevant
            }
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            # Default to RELEVANT on error to avoid false negatives
            logger.warning("⚠️ Error in relevance check - defaulting to RELEVANT")
            return {
                **state,
                "documents_relevant": True
            }

    
    def route_after_relevance(state: GraphState) -> str:
        """Route based on relevance"""
        if state.get("documents_relevant", False):
            logger.info("→ Routing to GENERATE")
            return "generate"
        else:
            logger.info("→ Routing to OUT_OF_SCOPE")
            return "out_of_scope"
    
    def generate_answer(state: GraphState) -> GraphState:
        """Generate answer from relevant documents"""
        query = state["query"]
        documents = state["documents"]
        intent_info = state.get("intent_info", "")
        
        logger.info("\nSTEP 6: ANSWER GENERATION")
        logger.info("="*80)
        logger.info(f"Documents for context: {len(documents)}")
        
        # Combine documents
        context = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{doc}" 
            for i, doc in enumerate(documents)
        ])
        
        logger.info(f"Context size: {len(context)} characters")
        
        # Determine answer type
        answer_type = "explanation"
        if "ANSWER_TYPE:" in intent_info:
            answer_type = intent_info.split("ANSWER_TYPE:")[1].split("\n")[0].strip().lower()
        
        # Select appropriate prompt based on answer type
        if answer_type in ["count", "list"]:
            logger.info(f"Using {answer_type.upper()} prompt template")
            rag_prompt = ChatPromptTemplate.from_template(
                """You are a helpful AI assistant. Answer based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
- Answer naturally without mentioning "according to documents" or "sources"
- If asked "how many", count ALL items carefully and provide exact number
- If asked to list, enumerate ALL items found
- Be comprehensive and specific
- Use natural conversational tone

Context:
{context}

Question: {question}

Provide complete answer with count/list:"""
            )
        else:
            logger.info("Using STANDARD prompt template")
            rag_prompt = ChatPromptTemplate.from_template(
                """You are a helpful AI assistant. Answer based ONLY on the provided context.

IMPORTANT:
- Answer naturally in conversational tone
- Strictly Answer to Question alone,Dont provide additional information that unrelated to question
- Don't mention sources or documents
- Provide clear, complete information
- If information is in context, answer confidently

Context:
{context}

Question: {question}

Answer:"""
            )
        
        try:
            response = llm.invoke(
                rag_prompt.format(context=context, question=query)
            )
            
            answer = response.content
            
            logger.info(f"Answer generated: {len(answer)} characters")
            logger.info(f"Preview: {answer[:200]}...")
            logger.info("="*80)
            
            return {
                **state,
                "generation": answer,
                "messages": state["messages"] + [{"role": "assistant", "content": answer}]
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            error_msg = "I apologize, but I encountered an error generating the response. Please try again."
            return {
                **state,
                "generation": error_msg,
                "messages": state["messages"] + [{"role": "assistant", "content": error_msg}]
            }
    
    def handle_out_of_scope(state: GraphState) -> GraphState:
        """Handle queries outside document scope"""
        logger.info("\nSTEP 6: OUT-OF-SCOPE HANDLER")
        logger.info("="*80)
        
        out_of_scope_message = (
            "I don't have information about that in my knowledge base. "
            "I can only answer questions about the documents that have been uploaded. "
            "Please ask questions related to the available documents, such as:\n"
            "- Company information\n"
            "- Branch locations\n"
            "- Products and services\n"
            "- Policies and procedures"
        )
        
        logger.info("Returning out-of-scope message")
        logger.info("="*80)
        
        return {
            **state,
            "generation": out_of_scope_message,
            "messages": state["messages"] + [{"role": "assistant", "content": out_of_scope_message}]
        }
    
    # Build the workflow
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("contextualize", add_conversational_context)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("reformulate", reformulate_query)
    workflow.add_node("retrieve", retrieve_with_intent)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("out_of_scope", handle_out_of_scope)
    
    # Add edges
    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "classify_intent")
    workflow.add_edge("classify_intent", "reformulate")
    workflow.add_edge("reformulate", "retrieve")
    workflow.add_edge("retrieve", "check_relevance")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "generate": "generate",
            "out_of_scope": "out_of_scope"
        }
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("out_of_scope", END)

    # Add memory
    memory = MemorySaver()
    
    # Compile
    agent = workflow.compile(checkpointer=memory)
    
    logger.info("✓ RAG Agent compiled successfully with natural language handling")
    
    
    return agent
