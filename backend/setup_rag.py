"""
Quick setup script for enabling high-quality RAG mode with FREE Groq API
"""
import os
import sys

def check_api_key():
    """Check if API key is set"""
    openai_key = os.environ.get('OPENAI_API_KEY')
    groq_key = os.environ.get('GROQ_API_KEY')
    
    if openai_key:
        print("✅ OpenAI API key found")
        return 'openai', openai_key
    elif groq_key:
        print("✅ Groq API key found")
        return 'groq', groq_key
    else:
        print("❌ No API key found")
        return None, None

def show_instructions():
    """Show setup instructions"""
    print("\n" + "="*70)
    print("🚀 RAG SETUP - Get FREE High-Quality Generation")
    print("="*70)
    
    print("\n📝 OPTION 1: Groq (RECOMMENDED - FREE & UNLIMITED)")
    print("-" * 70)
    print("1. Visit: https://console.groq.com/")
    print("2. Sign up (free, no credit card)")
    print("3. Go to 'API Keys' section")
    print("4. Create new key, copy it")
    print("5. Set environment variable:")
    print("   PowerShell: $env:GROQ_API_KEY = 'gsk_your-key-here'")
    print("   CMD: set GROQ_API_KEY=gsk_your-key-here")
    print("\n6. Install Groq: pip install groq")
    print("7. Restart this script")
    
    print("\n📝 OPTION 2: OpenAI ($5 Free Credits)")
    print("-" * 70)
    print("1. Visit: https://platform.openai.com/signup")
    print("2. Sign up (requires phone number)")
    print("3. Go to: https://platform.openai.com/api-keys")
    print("4. Create new secret key, copy it")
    print("5. Set environment variable:")
    print("   PowerShell: $env:OPENAI_API_KEY = 'sk-your-key-here'")
    print("   CMD: set OPENAI_API_KEY=sk-your-key-here")
    print("7. Restart this script")
    
    print("\n" + "="*70)
    print("💡 TIP: Groq is faster and FREE, perfect for testing!")
    print("="*70 + "\n")

def test_generation():
    """Test RAG generation with API key"""
    provider, key = check_api_key()
    
    if not provider:
        show_instructions()
        return False
    
    print(f"\n🔥 Testing {provider.upper()} generation...")
    
    try:
        if provider == 'openai':
            from ml_model_service_rag import RAGMLService
            service = RAGMLService(use_openai=True)
        else:
            # For Groq, we'd need to modify the service
            print("⚠️ Groq support requires modifying ml_model_service_rag.py")
            print("   See RAG_SETUP_GUIDE.md for instructions")
            return False
        
        # Test generation
        test_draft = "AI is transforming software development with code generation tools"
        
        print("\n" + "="*70)
        print(f"TEST DRAFT: {test_draft}")
        print("="*70)
        
        print("\n🔥 ENGAGEMENT HOOK:")
        print("-" * 70)
        result = service.generate_hook(test_draft)
        print(result['post'])
        
        print("\n✂️ CONCISE VERSION:")
        print("-" * 70)
        result = service.generate_concise(test_draft)
        print(result['post'])
        
        print("\n✨ PROFESSIONAL REPHRASE:")
        print("-" * 70)
        result = service.generate_rephrased(test_draft)
        print(result['post'])
        
        print("\n" + "="*70)
        print("✅ High-quality mode is working!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure API key is valid")
        print("2. Check internet connection")
        print("3. Verify openai package is installed: pip install openai")
        return False

if __name__ == "__main__":
    print("="*70)
    print("🎉 LinkedIn Post Optimizer - RAG Setup")
    print("="*70)
    
    success = test_generation()
    
    if not success:
        print("\n📚 For detailed instructions, see: RAG_SETUP_GUIDE.md")
        sys.exit(1)
    else:
        print("\n🚀 You can now start the backend:")
        print("   C:/Users/adity/Downloads/VS_Code/python.exe backend/app.py")
        sys.exit(0)
