"""
RAG-Enhanced ML Model Service for LinkedIn Post Optimization
Uses ChromaDB vector search + Groq (FREE Llama-3.1) or OpenAI GPT-3.5-turbo
"""
import os
from typing import Dict, List
from rag_service import RAGService

# Try Groq first (FREE!), fallback to OpenAI
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RAGMLService:
    def __init__(self):
        """Initialize RAG-enhanced ML service with Groq or OpenAI"""
        print("Initializing RAG-enhanced ML service...")
        
        # Initialize RAG service
        self.rag = RAGService()
        
        # Check if vector DB is empty, load data if needed
        if self.rag.collection.count() == 0:
            print("Vector database empty, loading training data...")
            self.rag.load_from_json("data/full_dataset.json")
        else:
            print(f"Vector database ready with {self.rag.collection.count()} posts.")
        
        # Try to initialize Groq (FREE!)
        groq_key = os.environ.get('GROQ_API_KEY')
        openai_key = os.environ.get('OPENAI_API_KEY')
        
        self.client = None
        self.provider = None
        
        if GROQ_AVAILABLE and groq_key:
            self.client = Groq(api_key=groq_key)
            self.provider = 'groq'
            print("✅ Groq Llama-3.1-70b enabled (FREE & FAST!)")
        elif OPENAI_AVAILABLE and openai_key:
            self.client = OpenAI(api_key=openai_key)
            self.provider = 'openai'
            print("✅ OpenAI GPT-3.5-turbo enabled")
        else:
            print("⚠️ No API key found!")
            print("   Get FREE Groq key: https://console.groq.com/")
            print("   Or OpenAI key: https://platform.openai.com/")
            print("   Falling back to pattern-based generation...")
    
    def generate_with_gpt(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text using Groq or OpenAI"""
        if not self.client:
            return None
        
        try:
            if self.provider == 'groq':
                # Groq Llama-3.1 (FREE, FAST!)
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Updated model name
                    messages=[
                        {"role": "system", "content": "You are a LinkedIn post optimization expert. Generate engaging, professional LinkedIn posts with specific examples, proper emojis (max 3), and relevant hashtags (2-3 max)."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                # OpenAI GPT-3.5
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a LinkedIn post optimization expert. Generate engaging, professional LinkedIn posts."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"{self.provider.upper()} API error: {e}")
            return None
    
    def create_hook_prompt(self, draft: str, similar_posts: List[Dict]) -> str:
        """Create prompt for engagement hook generation"""
        examples = "\n\n".join([
            f"Example {i+1}:\n{post['post']}"
            for i, post in enumerate(similar_posts[:3])
        ])
        
        prompt = f"""Create an engaging LinkedIn post with a strong hook that captures attention.

SIMILAR HIGH-PERFORMING POSTS FOR REFERENCE:
{examples}

USER DRAFT:
{draft}

REQUIREMENTS:
1. Start with an attention-grabbing hook (question, surprising stat, or bold statement)
2. Use max 2-3 contextual emojis (avoid overuse)
3. Add specific examples or data points (not generic statements)
4. Include a clear call-to-action or thought-provoking question at the end
5. Keep it concise but impactful (100-150 words)
6. Use relevant hashtags (2-3 max) that are specific to the topic
7. AVOID generic hashtags like #SEO, #Advisor, #Wedding, #Marketing, #Business

Generate an optimized LinkedIn post:"""
        return prompt
    
    def create_concise_prompt(self, draft: str, similar_posts: List[Dict]) -> str:
        """Create prompt for concise version generation"""
        examples = "\n\n".join([
            f"Example {i+1}:\n{post['post']}"
            for i, post in enumerate(similar_posts[:3])
        ])
        
        prompt = f"""Create a concise, punchy LinkedIn post that gets straight to the point.

SIMILAR HIGH-PERFORMING POSTS FOR REFERENCE:
{examples}

USER DRAFT:
{draft}

REQUIREMENTS:
1. Keep it ultra-concise (50-80 words max)
2. Use 1-2 relevant emojis for emphasis
3. Focus on the core message only
4. End with a question or CTA to drive engagement
5. Use 1-2 hashtags max

Generate a concise LinkedIn post:"""
        return prompt
    
    def create_rephrased_prompt(self, draft: str, similar_posts: List[Dict]) -> str:
        """Create prompt for professional rephrasing"""
        examples = "\n\n".join([
            f"Example {i+1}:\n{post['post']}"
            for i, post in enumerate(similar_posts[:3])
        ])
        
        prompt = f"""Rephrase this draft into a professional, polished LinkedIn post.

SIMILAR HIGH-PERFORMING POSTS FOR REFERENCE:
{examples}

USER DRAFT:
{draft}

REQUIREMENTS:
1. Maintain professional yet approachable tone
2. Add storytelling elements if applicable
3. Use 2-3 strategic emojis (not excessive)
4. Include 2-3 relevant, specific hashtags (not generic ones like #SEO, #Advisor)
5. End with engagement driver (question, CTA, or insight)
6. 100-200 words

Generate a rephrased LinkedIn post:"""
        return prompt
    
    def post_process(self, text: str, style: str) -> str:
        """Post-process generated text to ensure quality"""
        if not text:
            return ""
        
        # Remove excessive emojis (max 3)
        emoji_count = sum(1 for char in text if ord(char) > 127000)
        if emoji_count > 3:
            # Simple approach: keep first 3 emojis
            pass
        
        # Ensure hashtags are relevant (basic check)
        generic_hashtags = ['#SEO', '#Advisor', '#Wedding', '#Marketing', '#Business']
        for tag in generic_hashtags:
            if tag in text and tag not in text[:50]:  # Don't remove if in first line
                text = text.replace(tag, '')
        
        # Ensure there's a question or CTA at the end
        if style == 'hook' and not any(text.endswith(x) for x in ['?', '!', '👇', '💬']):
            text += "\n\nWhat are your thoughts?"
        
        return text.strip()
    
    def generate_hook(self, draft: str) -> Dict[str, str]:
        """Generate engagement hook version using RAG + LLM"""
        # Find similar posts for context
        similar_posts = self.rag.find_similar_posts(draft, n_results=3)
        
        if self.client and similar_posts:
            # Use Groq/GPT with RAG context
            prompt = self.create_hook_prompt(draft, similar_posts)
            generated = self.generate_with_gpt(prompt, max_tokens=200, temperature=0.7)
            
            if generated:
                post = self.post_process(generated, 'hook')
                return {
                    'style': '🔥 Engagement Hook',
                    'post': post
                }
        
        # Fallback to pattern-based generation
        return {
            'style': '🔥 Engagement Hook',
            'post': f"🚀 {draft}\n\nWhat's your experience with this? Let me know in the comments! 💬"
        }
    
    def generate_concise(self, draft: str) -> Dict[str, str]:
        """Generate concise version using RAG + LLM"""
        similar_posts = self.rag.find_similar_posts(draft, n_results=3)
        
        if self.client and similar_posts:
            prompt = self.create_concise_prompt(draft, similar_posts)
            generated = self.generate_with_gpt(prompt, max_tokens=100, temperature=0.6)
            
            if generated:
                post = self.post_process(generated, 'concise')
                return {
                    'style': '✂️ Concise Version',
                    'post': post
                }
        
        # Fallback
        return {
            'style': '✂️ Concise Version',
            'post': f"💡 {draft}"
        }
    
    def generate_rephrased(self, draft: str) -> Dict[str, str]:
        """Generate professionally rephrased version using RAG + LLM"""
        similar_posts = self.rag.find_similar_posts(draft, n_results=3)
        
        if self.client and similar_posts:
            prompt = self.create_rephrased_prompt(draft, similar_posts)
            generated = self.generate_with_gpt(prompt, max_tokens=250, temperature=0.7)
            
            if generated:
                post = self.post_process(generated, 'rephrased')
                return {
                    'style': '✨ Professional Rephrase',
                    'post': post
                }
        
        # Fallback
        return {
            'style': '✨ Professional Rephrase',
            'post': f"📈 {draft}"
        }


# Singleton instance
_service = None

def get_service():
    """Get or create ML service instance"""
    global _service
    if _service is None:
        _service = RAGMLService()
    return _service


# API functions for Flask integration
def generate_hook(draft: str) -> Dict[str, str]:
    service = get_service()
    return service.generate_hook(draft)

def generate_concise(draft: str) -> Dict[str, str]:
    service = get_service()
    return service.generate_concise(draft)

def generate_rephrased(draft: str) -> Dict[str, str]:
    service = get_service()
    return service.generate_rephrased(draft)


# Test the service
if __name__ == "__main__":
    print("Testing RAG-enhanced ML service...")
    service = RAGMLService()
    
    test_draft = "AI is transforming software development with code generation tools"
    
    print("\n" + "="*70)
    print("TEST DRAFT:", test_draft)
    print("="*70)
    
    print("\n1. ENGAGEMENT HOOK:")
    result = service.generate_hook(test_draft)
    print(f"Style: {result['style']}")
    print(f"Post:\n{result['post']}\n")
    
    print("2. CONCISE VERSION:")
    result = service.generate_concise(test_draft)
    print(f"Style: {result['style']}")
    print(f"Post:\n{result['post']}\n")
    
    print("3. PROFESSIONAL REPHRASE:")
    result = service.generate_rephrased(test_draft)
    print(f"Style: {result['style']}")
    print(f"Post:\n{result['post']}\n")
