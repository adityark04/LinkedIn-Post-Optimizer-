"""
Enhanced LinkedIn Scraper - Collect 100+ posts using BeautifulSoup4
"""

import json
import os
import re
import time
import random
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests


class LinkedInBulkScraper:
    """Scraper to collect 100+ LinkedIn posts"""
    
    def __init__(self, headless=True):
        self.headless = headless
        self.posts = []
        
    def setup_selenium_driver(self):
        """Setup Selenium Chrome driver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def scrape_with_selenium(self, url, driver):
        """Scrape single post with Selenium + BeautifulSoup"""
        try:
            print(f"  Fetching: {url[:60]}...")
            driver.get(url)
            
            # Wait for content to load
            time.sleep(random.uniform(3, 5))
            
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Try multiple selectors for post content
            selectors = [
                '.feed-shared-update-v2__description',
                '.feed-shared-text',
                '.update-components-text',
                'div[dir="ltr"]',
                '.feed-shared-inline-show-more-text',
                'article',
            ]
            
            text = None
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    # Get text from all matching elements
                    for elem in elements:
                        content = elem.get_text(strip=True)
                        if content and len(content) > 50:  # Min 50 chars
                            text = content
                            break
                if text:
                    break
            
            # Fallback: search for any div with substantial text
            if not text:
                all_divs = soup.find_all('div')
                for div in all_divs:
                    content = div.get_text(strip=True)
                    if len(content) > 100 and len(content) < 2000:
                        text = content
                        break
            
            if text:
                # Clean text
                text = re.sub(r'\s+', ' ', text).strip()
                return {'text': text, 'url': url, 'method': 'selenium'}
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Selenium error: {e}")
            return None
    
    def scrape_with_requests(self, url):
        """Scrape with requests + BeautifulSoup (fallback)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find post content
            selectors = [
                '.feed-shared-update-v2__description',
                '.feed-shared-text',
                'article',
                'div[dir="ltr"]',
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    for elem in elements:
                        text = elem.get_text(strip=True)
                        if text and len(text) > 50:
                            text = re.sub(r'\s+', ' ', text).strip()
                            return {'text': text, 'url': url, 'method': 'requests'}
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Requests error: {e}")
            return None
    
    def generate_sample_posts(self, count=100):
        """Generate diverse sample LinkedIn posts for training"""
        print(f"📝 Generating {count} diverse sample posts...")
        
        # Templates for different post types
        templates = {
            'achievement': [
                "Excited to share that I've been promoted to {role} at {company}! Grateful for the journey and the amazing team. {tags}",
                "Just completed {achievement}! It took {time} but the learning was worth it. {tags}",
                "Proud to announce {milestone}! Thank you to everyone who supported me along the way. {tags}",
            ],
            'learning': [
                "5 key lessons I learned from {experience}: 1) {lesson1} 2) {lesson2} 3) {lesson3} 4) {lesson4} 5) {lesson5}. {tags}",
                "After {years} years in {field}, here's what I wish I knew earlier: {insight}. {tags}",
                "Top 3 mistakes I made in {area} and how I fixed them: {mistake1}, {mistake2}, {mistake3}. {tags}",
            ],
            'tips': [
                "Want to improve your {skill}? Try these 3 strategies: 1) {tip1} 2) {tip2} 3) {tip3}. What works for you? {tags}",
                "The best advice I ever received about {topic}: {advice}. Game changer! {tags}",
                "{number} habits that transformed my {area}: {habit1}, {habit2}, {habit3}. Start small! {tags}",
            ],
            'story': [
                "Two years ago, I was {past_situation}. Today, I'm {current_situation}. The journey taught me {lesson}. {tags}",
                "When I started {activity}, everyone said {objection}. But I persisted, and now {result}. {tags}",
                "Failed {number} times before {success}. Each failure taught me something valuable. Keep going! {tags}",
            ],
            'insights': [
                "The future of {industry} is here: {trend1}, {trend2}, and {trend3} are changing everything. {tags}",
                "Why {topic} matters more than ever: {reason1}, {reason2}, {reason3}. Thoughts? {tags}",
                "Unpopular opinion: {opinion}. Here's why: {explanation}. {tags}",
            ],
            'announcement': [
                "Big news! {announcement}. Can't wait to share more details soon. {tags}",
                "Thrilled to announce {news}! This is just the beginning. {tags}",
                "After {time} of hard work, we're launching {product}! Check it out: {details}. {tags}",
            ],
        }
        
        # Replacement values
        replacements = {
            'role': ['Senior Engineer', 'Team Lead', 'Product Manager', 'Director', 'VP'],
            'company': ['Microsoft', 'Google', 'Amazon', 'Meta', 'Apple', 'a fast-growing startup'],
            'achievement': ['my AWS certification', 'a major project', 'my MBA', 'a successful product launch'],
            'time': ['6 months', '1 year', '2 years', '3 months'],
            'milestone': ['our team hit 1M users', 'we closed a major deal', "I published my first book"],
            'experience': ['leading a team', 'building products', 'scaling systems', 'managing projects'],
            'lesson1': ['Trust your team', 'Communication is key', 'Fail fast, learn faster'],
            'lesson2': ['Focus on impact', 'Listen more than you speak', 'Data beats opinions'],
            'lesson3': ['Celebrate small wins', 'Build relationships', 'Stay curious'],
            'lesson4': ['Take calculated risks', 'Learn continuously', 'Seek feedback'],
            'lesson5': ['Balance is crucial', 'Help others grow', 'Stay humble'],
            'years': ['5', '10', '3', '7'],
            'field': ['tech', 'product management', 'engineering', 'data science', 'leadership'],
            'insight': ['focus on learning, not just doing', 'relationships matter more than skills', 'consistency beats intensity'],
            'area': ['my career', 'product development', 'team building', 'project management'],
            'mistake1': ['not asking for help sooner', 'overcomplicating solutions', 'ignoring feedback'],
            'mistake2': ['poor time management', 'not setting boundaries', 'avoiding difficult conversations'],
            'mistake3': ['trying to please everyone', 'not delegating enough', 'perfectionism'],
            'skill': ['communication', 'leadership', 'coding', 'productivity', 'networking'],
            'tip1': ['Practice daily', 'Find a mentor', 'Learn from failures'],
            'tip2': ['Set clear goals', 'Get feedback', 'Stay consistent'],
            'tip3': ['Teach others', 'Read widely', 'Build projects'],
            'topic': ['career growth', 'leadership', 'work-life balance', 'productivity'],
            'advice': ['focus on what you can control', 'your network is your net worth', 'done is better than perfect'],
            'number': ['3', '5', '7', '10'],
            'habit1': ['Morning routine', 'Daily reading', 'Exercise'],
            'habit2': ['Journaling', 'Deep work blocks', 'Regular breaks'],
            'habit3': ['Continuous learning', 'Networking', 'Meditation'],
            'past_situation': ['struggling to find my path', 'working 80-hour weeks', 'afraid to take risks'],
            'current_situation': ['leading a team of 20', 'running my own company', 'mentoring others'],
            'lesson': ['persistence pays off', 'your mindset shapes your reality', 'help is always available'],
            'activity': ['my career', 'this project', 'learning to code', 'building products'],
            'objection': ['"it won\'t work"', '"you\'re too late"', '"stick to what you know"'],
            'result': ['"we have 100K users"', '"I landed my dream job"', '"we\'re profitable"'],
            'success': ['landing this role', 'launching successfully', 'making it work'],
            'industry': ['AI', 'tech', 'remote work', 'software development', 'data science'],
            'trend1': ['AI automation', 'Remote-first culture', 'No-code tools'],
            'trend2': ['API-first design', 'Sustainable tech', 'Edge computing'],
            'trend3': ['Personalization', 'Real-time collaboration', 'Privacy-first'],
            'reason1': ['it drives innovation', 'it saves time', 'it scales impact'],
            'reason2': ['everyone can benefit', 'the ROI is clear', 'it\'s more accessible'],
            'reason3': ['the technology is mature', 'competition demands it', 'customers expect it'],
            'opinion': ['meetings are overrated', 'side projects > certifications', 'soft skills > hard skills'],
            'explanation': ['async communication is more efficient', 'you learn by doing', 'they unlock opportunities'],
            'announcement': ['I\'m joining a new company', 'we\'re hiring', 'our product is live'],
            'news': ['our Series A funding', 'a partnership with XYZ', 'my new role'],
            'product': ['our new AI tool', 'an open-source library', 'a course for developers'],
            'details': ['link in comments', 'DM for early access', 'public beta starts next week'],
            'tags': ['#Tech #Career', '#Leadership #Growth', '#Productivity #Success', '#AI #Innovation', 
                    '#CareerAdvice #Professional', '#TechLife #Engineering', '#Startup #Entrepreneur',
                    '#Learning #Development', '#WorkCulture #TeamWork'],
        }
        
        posts = []
        post_types = list(templates.keys())
        
        for i in range(count):
            # Pick random template
            post_type = random.choice(post_types)
            template = random.choice(templates[post_type])
            
            # Fill in template
            post = template
            for key, values in replacements.items():
                if '{' + key + '}' in post:
                    post = post.replace('{' + key + '}', random.choice(values))
            
            posts.append({
                'text': post,
                'url': f'generated_{i+1}',
                'method': 'generated',
                'type': post_type
            })
        
        print(f"✓ Generated {len(posts)} diverse posts")
        return posts
    
    def save_posts(self, filename='data/scraped_posts.json'):
        """Save scraped posts to JSON"""
        os.makedirs('data', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.posts, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(self.posts)} posts to {filename}")


def convert_to_training_format(posts: List[Dict]) -> List[Dict]:
    """Convert scraped posts to training format"""
    print("\n🔄 Converting posts to training format...")
    training_data = []
    
    for post in posts:
        text = post['text']
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
        
        # Create draft (first sentence or up to 100 chars)
        draft = sentences[0] if len(sentences[0]) < 100 else text[:100].rsplit(' ', 1)[0] + '...'
        
        # Create hook (full text with emoji if not present)
        hook = text
        if not any(char in text for char in ['🚀', '💡', '✨', '🎯', '📈', '💪']):
            emojis = ['🚀', '💡', '✨', '🎯', '📈', '💪', '🔥', '⚡', '🌟']
            hook = random.choice(emojis) + ' ' + text
        
        # Create concise (first 2 sentences or 150 chars)
        if len(sentences) > 1:
            concise = ' '.join(sentences[:2])
        else:
            concise = text[:150].rsplit(' ', 1)[0] + '...' if len(text) > 150 else text
        
        # Create rephrased (full text with hashtags if not present)
        rephrased = text
        if '#' not in text:
            # Extract hashtags from original or generate
            hashtags = re.findall(r'#\w+', text)
            if not hashtags:
                # Generate relevant hashtags
                common_tags = ['#LinkedIn', '#Professional', '#Career', '#Growth', '#Success', '#Learning']
                hashtags = random.sample(common_tags, 2)
            rephrased = text + ' ' + ' '.join(hashtags)
        
        training_data.append({
            'draft': draft,
            'hook': hook,
            'concise': concise,
            'rephrased': rephrased,
            'source': post.get('method', 'scraped'),
            'original_url': post.get('url', 'unknown')
        })
    
    print(f"✓ Converted {len(training_data)} posts to training format")
    return training_data


def main():
    print("="*60)
    print("LinkedIn Bulk Scraper - Collect 100+ Posts")
    print("="*60)
    
    scraper = LinkedInBulkScraper(headless=True)
    
    print("\nChoose scraping method:")
    print("1. Provide LinkedIn post URLs (manual collection)")
    print("2. Generate 100+ diverse sample posts (recommended)")
    print("3. Combination (URLs + generated samples)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    all_posts = []
    
    if choice in ['1', '3']:
        # Manual URL collection
        print("\nEnter LinkedIn post URLs (one per line, empty line to finish):")
        urls = []
        while True:
            url = input().strip()
            if not url:
                break
            urls.append(url)
        
        if urls:
            print(f"\n🌐 Scraping {len(urls)} URLs...")
            driver = scraper.setup_selenium_driver()
            
            for i, url in enumerate(urls, 1):
                print(f"[{i}/{len(urls)}]", end=' ')
                post = scraper.scrape_with_selenium(url, driver)
                if post:
                    all_posts.append(post)
                    print("  ✓ Success")
                else:
                    print("  ✗ Failed")
                
                # Polite delay
                time.sleep(random.uniform(2, 4))
            
            driver.quit()
    
    if choice in ['2', '3']:
        # Generate sample posts
        target_count = 100
        if choice == '3' and all_posts:
            target_count = max(20, 100 - len(all_posts))
        
        generated = scraper.generate_sample_posts(target_count)
        all_posts.extend(generated)
    
    if not all_posts:
        print("❌ No posts collected")
        return
    
    # Save scraped posts
    scraper.posts = all_posts
    scraper.save_posts()
    
    # Convert to training format
    training_data = convert_to_training_format(all_posts)
    
    with open('data/scraped_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to data/scraped_training_data.json")
    
    print("\n" + "="*60)
    print("✅ Scraping Complete!")
    print(f"   Scraped: {len(all_posts)} posts")
    print(f"   Output: data/scraped_posts.json")
    print(f"   Training: data/scraped_training_data.json")
    print("="*60)
    print("\n🚀 Next steps:")
    print("   1. Run: python merge_scraped_data.py")
    print("   2. Run: python train_models.py")


if __name__ == '__main__':
    main()
