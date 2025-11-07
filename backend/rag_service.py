"""
RAG Service for LinkedIn Post Optimization
Uses ChromaDB vector database + sentence-transformers for similarity search
"""
import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class RAGService:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize RAG service with ChromaDB and sentence transformer"""
        # Initialize embedding model
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="linkedin_posts",
            metadata={"description": "High-quality LinkedIn posts for RAG"}
        )
        
        print(f"RAG Service initialized. Collection has {self.collection.count()} posts.")
    
    def embed_text(self, text: str):
        """Generate embedding for a text"""
        return self.embedding_model.encode(text).tolist()
    
    def add_posts(self, posts: List[Dict]):
        """Add posts to vector database"""
        if not posts:
            print("No posts to add.")
            return
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, post in enumerate(posts):
            # Try multiple field names (post, hook, rephrased, concise)
            post_text = post.get('post') or post.get('hook') or post.get('rephrased') or post.get('concise') or post.get('draft', '')
            if not post_text:
                continue
            
            documents.append(post_text)
            embeddings.append(self.embed_text(post_text))
            
            # Determine post type from available fields
            post_type = 'unknown'
            if 'hook' in post:
                post_type = 'hook'
            elif 'concise' in post:
                post_type = 'concise'
            elif 'rephrased' in post:
                post_type = 'rephrased'
            
            # Store metadata
            metadata = {
                'type': post_type,
                'length': len(post_text.split()),
                'has_emoji': '🔥' in post_text or '💡' in post_text or '🚀' in post_text,
                'has_hashtag': '#' in post_text,
                'source': post.get('source', 'training')
            }
            metadatas.append(metadata)
            ids.append(f"post_{i}")
        
        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(documents)} posts to vector database.")
    
    def find_similar_posts(self, query: str, n_results: int = 3, post_type = None):
        """Find similar posts using vector similarity search"""
        query_embedding = self.embed_text(query)
        
        # Build where filter if post_type specified
        where_filter = None
        if post_type:
            where_filter = {"type": {"$eq": post_type}}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )
        except:
            # If filter fails, try without filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        
        similar_posts = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                similar_posts.append({
                    'post': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0
                })
        
        return similar_posts
    
    def load_from_json(self, json_path: str):
        """Load posts from JSON file and add to vector database"""
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict) and 'posts' in data:
            posts = data['posts']
        else:
            print("Unknown JSON structure")
            return
        
        self.add_posts(posts)
    
    def get_stats(self):
        """Get database statistics"""
        count = self.collection.count()
        return {
            'total_posts': count,
            'embedding_model': 'all-MiniLM-L6-v2',
            'vector_dimension': 384
        }


# Utility function for testing
if __name__ == "__main__":
    # Initialize RAG service
    rag = RAGService()
    
    # Load existing training data
    print("\nLoading training data...")
    rag.load_from_json("data/full_dataset.json")
    
    # Test similarity search
    print("\nTesting similarity search...")
    test_query = "AI is transforming the future of work and productivity"
    results = rag.find_similar_posts(test_query, n_results=3)
    
    print(f"\nQuery: {test_query}\n")
    print("Similar posts found:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. (Distance: {result['distance']:.3f})")
        print(f"   Type: {result['metadata'].get('type', 'unknown')}")
        print(f"   Post: {result['post'][:150]}...")
    
    # Print stats
    print(f"\nDatabase stats: {rag.get_stats()}")
