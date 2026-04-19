import os
import asyncio
import random
from langchain_core.tools import tool
from openai import AsyncOpenAI
from app.db.qdrant_client import search


@tool
async def search_knowledge_base(query: str) -> str:
    """
    Searches the ShopWave policy and FAQ knowledge base using semantic similarity.
    Use this to look up:
      - Return window policies (e.g. 'return policy for electronics')
      - Warranty rules and coverage (e.g. 'warranty policy for smart watches')
      - Escalation guidelines (e.g. 'when to escalate a ticket')
      - Refund eligibility rules (e.g. 'refund policy for damaged items')
      - General FAQs and support procedures

    Input should be a specific, descriptive search query.
    Returns the top 3 most relevant policy excerpts from the knowledge base.
    If no relevant policies are found, returns a 'not found' message.
    """
    # Small simulated latency to ensure async tasks yield control properly (Hackathon concurrency test)
    await asyncio.sleep(random.uniform(0.1, 0.2))

    if not query or not query.strip():
        return "KnowledgeBaseError: No query provided. Please supply a specific search query."

    query = query.strip()

    try:
        client = AsyncOpenAI()
        
        response = await client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_vector = response.data[0].embedding
        
        results = await asyncio.to_thread(search, query_vector=query_vector, top_k=3)
        
        if not results:
            return "No relevant policies found in the knowledge base. If unsure, escalate the ticket."
        
        formatted_results = "--- KNOWLEDGE BASE RESULTS ---\n\n"
        for i, res in enumerate(results):
            metadata = res.payload.get("metadata", {})
            
            section = metadata.get("Header 2", metadata.get("Section", "Policy Section"))
            subsection = metadata.get("Header 3", metadata.get("Subsection", ""))
            
            context = f"[{section}"
            if subsection:
                context += f" > {subsection}"
            context += "]"

            formatted_results += f"Result {i+1} {context}:\n{res.payload['content']}\n\n"
            
        return formatted_results

    except Exception as e:
        return f"KnowledgeBaseError: Failed to retrieve policies due to {str(e)}. Please try a different query or escalate."