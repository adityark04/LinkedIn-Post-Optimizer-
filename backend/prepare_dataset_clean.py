"""
Clean dataset preparation script with 40 high-quality examples.
This provides enough data for the models to learn patterns without overfitting.
"""

import json
import os
import random

def create_linkedin_dataset():
    """Generate 40 quality LinkedIn post examples for training."""
    
    # 40 carefully crafted examples with realistic LinkedIn post variations
    data = [
        # Example 1
        {
            "draft": "I won a contest at JP Morgan",
            "hook": "Never thought I would say this but I just won the JP Morgan coding challenge! Months of preparation finally paid off. The competition tested algorithmic thinking and problem solving under pressure.",
            "concise": "Won JP Morgan coding contest! Months of LeetCode practice finally paid off. Grateful for this opportunity.",
            "rephrased": "Thrilled to announce I won the JP Morgan coding competition! This achievement represents months of dedicated practice and strategic preparation. Grateful for this recognition."
        },
        # Example 2
        {
            "draft": "Launched our new mobile app today",
            "hook": "From idea to app store in just 3 months! Today we are officially live on iOS and Android. The journey of building our mobile app taught me more than 4 years of college.",
            "concise": "Mobile app launched today! 3 months from concept to production. Team effort made this possible.",
            "rephrased": "Exciting milestone: Our mobile app is officially live! Three months ago this was just an idea. Today it serves users worldwide. Huge shout-out to our incredible team."
        },
        # Example 3
        {
            "draft": "Just got promoted to Senior Engineer",
            "hook": "Career milestone unlocked! From intern to Senior Engineer in 2 years. This promotion represents continuous learning and embracing challenges outside my comfort zone.",
            "concise": "Promoted to Senior Engineer! 2 years of growth and pushing boundaries. Excited for what comes next.",
            "rephrased": "Career milestone: Promoted to Senior Software Engineer! In 2 years I grew from writing my first code to architecting systems serving millions. Growth happens outside your comfort zone."
        },
        # Example 4
        {
            "draft": "Completed my first marathon this weekend",
            "hook": "Just crossed the finish line of my first marathon! 26 miles taught me that mental resilience and persistence translate directly to software engineering. Whether debugging or pushing through mile 20 keep moving forward.",
            "concise": "First marathon complete! 26 miles taught me persistence translates everywhere in life.",
            "rephrased": "Marathon milestone achieved! Training for this race taught me discipline and mental resilience that directly apply to my engineering career. The principle is simple: keep moving forward."
        },
        # Example 5
        {
            "draft": "Our startup raised Series A funding",
            "hook": "We did it! $5M Series A is officially closed. But here is what nobody tells you about raising venture capital. Fundraising is a marathon not a sprint. Persistence and preparation are everything.",
            "concise": "Series A closed! $5M raised. Grateful to our investors and team. Now the real work begins.",
            "rephrased": "Major announcement: We successfully closed our $5M Series A funding round! This milestone validates our vision. Immense gratitude to investors who believe in our team."
        },
        # Example 6
        {
            "draft": "Published my first research paper on AI",
            "hook": "My first AI research paper is officially published! 18 months of work 47 rejections and one breakthrough moment. Research requires resilience as much as intelligence.",
            "concise": "First AI research paper published! 18 months of work validated. Excited to contribute to the field.",
            "rephrased": "Published! My first AI research paper is now live. After 18 months of experiments and 47 rejections our work on attention mechanisms is finally contributing to the field."
        },
        # Example 7
        {
            "draft": "Spoke at TechConf about cloud architecture",
            "hook": "Just stepped off stage at TechConf 2025! Speaking to 2000 developers about cloud architecture was surreal. The energy in that room was electric. Key themes: microservices cost optimization and reliability at scale.",
            "concise": "Spoke at TechConf 2025! Shared cloud architecture insights with 2000 developers. Incredible experience.",
            "rephrased": "What an experience speaking at TechConf 2025! Shared insights on cloud architecture optimization with 2000 developers. Honored to contribute to the community."
        },
        # Example 8
        {
            "draft": "Hit 10k users on our platform",
            "hook": "We hit 10000 users! Six months ago we had exactly 3 users. Reaching 10k validates product-market fit and proves solving real problems attracts real users.",
            "concise": "Hit 10000 users! From 3 to 10k in 6 months. Product-market fit achieved.",
            "rephrased": "We hit 10000 users! Six months ago we had 3 users. This milestone validates our product-market fit. To early stage founders: focus on value and iterate fast."
        },
        # Example 9
        {
            "draft": "Joined Meta as a software engineer",
            "hook": "New chapter begins! Joining Meta as a Software Engineer. The interview process was intense but my preparation strategy made the difference. It is achievable with structured preparation.",
            "concise": "Joined Meta as Software Engineer! New chapter begins. Ready to build at scale.",
            "rephrased": "Excited to join Meta as a Software Engineer! The interview process was challenging but structured preparation made it achievable. Happy to share resources that helped."
        },
        # Example 10
        {
            "draft": "Our team won the hackathon",
            "hook": "Champion team! We won the Global Hackathon 2025! 48 hours of coding 0 hours of sleep and one functioning MVP. Beyond the trophy we learned about rapid prototyping and team dynamics under pressure.",
            "concise": "Won Global Hackathon! 48 hours of intense building paid off. Team collaboration was key.",
            "rephrased": "We won the Global Hackathon 2025! 48 hours of intensive coding resulted in an AI-powered accessibility tool. Learned invaluable lessons about rapid prototyping and team dynamics."
        },
        # Continue with 30 more examples...
        # Example 11
        {
            "draft": "Achieved AWS certification",
            "hook": "AWS Solutions Architect certified! This exam tested everything: well-architected framework disaster recovery and cost optimization. Study advice: theory alone won't cut it build actual systems.",
            "concise": "AWS Solutions Architect certified! Challenging but worth it. Cloud skills leveled up.",
            "rephrased": "AWS Solutions Architect - Professional achieved! This exam tested comprehensive cloud knowledge. Resources that worked: whitepapers hands-on labs and real-world projects."
        },
        # Example 12
        {
            "draft": "Built a chatbot using GPT",
            "hook": "Production-ready chatbot deployed! Built with GPT-4 achieving 0.8 second response time and 94 percent user satisfaction. The gap between demo and production is massive.",
            "concise": "Production chatbot live! GPT-4 powered with 0.8s response time. AI in action.",
            "rephrased": "Chatbot deployed to production! Built with GPT-4 achieving excellent response time and user satisfaction. Architecture highlights: Redis caching async processing and monitoring."
        },
        # Example 13
        {
            "draft": "Reduced server costs by 60 percent",
            "hook": "Cost optimization success! Reduced our AWS bill from $50k to $20k monthly without sacrificing performance. Cloud costs can spiral but strategic optimization enables massive savings.",
            "concise": "Cut AWS costs 60 percent! From $50k to $20k monthly with no performance loss. Optimization wins.",
            "rephrased": "Cost optimization win: Reduced our AWS bill from $50000 to $20000 monthly! Achieved through right-sizing auto-scaling reserved instances and query optimization."
        },
        # Example 14
        {
            "draft": "Started a tech YouTube channel",
            "hook": "YouTube milestone: 1000 subscribers! Started documenting my coding journey 3 months ago. Creating content taught me about teaching communication and deep understanding.",
            "concise": "Hit 1000 YouTube subscribers! Tech content journey begins. More tutorials coming.",
            "rephrased": "Reached 1000 YouTube subscribers! Started creating tech content 3 months ago. Lessons learned: start before you are ready consistency beats perfection engage genuinely."
        },
        # Example 15
        {
            "draft": "Deployed ML model to production",
            "hook": "First ML model in production! Journey from Jupyter notebook to serving 10000 predictions daily. Real-world AI deployment is as much engineering as data science.",
            "concise": "ML model in production! From notebook to 10k predictions daily. Real-world deployment achieved.",
            "rephrased": "Machine learning model deployed! Now serving 10000 predictions daily in production. Key learnings: model versioning monitoring A/B testing are non-negotiable."
        },
        # Example 16
        {
            "draft": "Built AI recommendation system",
            "hook": "Recommendation engine live! Built collaborative filtering system improving user engagement by 45 percent. Machine learning drives personalization at scale.",
            "concise": "AI recommendation system deployed! User engagement up 45 percent. ML personalization works.",
            "rephrased": "Recommendation system launched! Our collaborative filtering engine increased user engagement 45 percent. Machine learning enables true personalization at scale."
        },
        # Example 17
        {
            "draft": "Created automated testing framework",
            "hook": "Testing framework shipped! Automated 80 percent of manual QA reducing release time from weeks to days. Quality automation accelerates delivery.",
            "concise": "Automated testing framework live! 80 percent of QA automated. Release time cut dramatically.",
            "rephrased": "Testing automation success! New framework automated 80 percent of manual QA. Result: release cycles reduced from weeks to days while improving quality."
        },
        # Example 18
        {
            "draft": "Developed REST API service",
            "hook": "API service deployed! Built scalable REST API handling 1 million requests daily. Proper architecture and caching enable massive throughput.",
            "concise": "REST API service live! Handling 1 million requests daily. Scalability achieved.",
            "rephrased": "REST API launched! Now serving 1 million requests daily with 99.9 percent uptime. Architecture: load balancing Redis caching and comprehensive monitoring."
        },
        # Example 19
        {
            "draft": "Built computer vision system",
            "hook": "Computer vision deployed! Object detection system achieving 95 percent accuracy in real-time. Deep learning transforms image processing.",
            "concise": "Computer vision system live! 95 percent accuracy in real-time detection. Deep learning works.",
            "rephrased": "Computer vision achievement! Object detection system deployed with 95 percent accuracy. Deep learning and optimization techniques enable real-time processing."
        },
        # Example 20
        {
            "draft": "Developed cloud storage solution",
            "hook": "Cloud storage launched! Built distributed file system handling petabytes of data. System design principles enable infinite scale.",
            "concise": "Cloud storage solution deployed! Handling petabytes of data. Distributed systems work.",
            "rephrased": "Cloud storage system launched! Distributed architecture now handles petabytes of data. Key: consistent hashing replication and distributed consensus."
        },
        # Example 21
        {
            "draft": "Mentored 5 junior developers this quarter",
            "hook": "Mentorship win! All 5 junior developers I mentored got promoted. My philosophy: provide guidance not solutions. Challenge assumptions celebrate wins create psychological safety.",
            "concise": "5 mentees promoted this quarter! Proud to support their growth. Mentorship matters.",
            "rephrased": "Proudest achievement: All 5 junior developers I mentored this quarter got promoted! Mentoring approach: guidance over solutions psychological safety growth mindset."
        },
        # Example 22
        {
            "draft": "Organized first company hackathon",
            "hook": "Hackathon success! Organized our first company-wide event with 100 participants. Result: 3 projects moved to production. Innovation thrives in creative environments.",
            "concise": "First company hackathon complete! 100 participants 3 projects to production. Innovation wins.",
            "rephrased": "Company hackathon organized! 100 participants collaborated and 3 winning projects are moving to production. Creating space for innovation drives real results."
        },
        # Example 23
        {
            "draft": "Improved team productivity by 40 percent",
            "hook": "Productivity transformation! Improved team output 40 percent through better processes and tools. Key changes: async communication automated workflows focused time blocks.",
            "concise": "Team productivity up 40 percent! Better processes and tools made the difference.",
            "rephrased": "Productivity milestone: Team output increased 40 percent! Achieved through async communication automated workflows and protected focus time. Process matters."
        },
        # Example 24
        {
            "draft": "Established code review process",
            "hook": "Code review process launched! Reduced bugs by 60 percent and knowledge sharing improved dramatically. Peer review elevates entire team.",
            "concise": "Code review process established! Bugs down 60 percent. Knowledge sharing improved.",
            "rephrased": "Code review system implemented! Result: 60 percent fewer bugs and significantly better knowledge sharing. Peer review elevates code quality and team expertise."
        },
        # Example 25
        {
            "draft": "Created onboarding program",
            "hook": "Onboarding program launched! New hires now productive in 2 weeks vs 2 months. Structured onboarding accelerates contribution and retention.",
            "concise": "Onboarding program created! New hire productivity in 2 weeks vs 2 months. Structure works.",
            "rephrased": "Onboarding transformation! New program gets engineers productive in 2 weeks instead of 2 months. Structured onboarding improves contribution and retention."
        },
        # Example 26
        {
            "draft": "AI will transform healthcare",
            "hook": "Healthcare transformation ahead! AI enables early disease detection personalized treatment and predictive care. The potential to save lives is unprecedented.",
            "concise": "AI transforming healthcare! Early detection personalized treatment predictive care. Lives will be saved.",
            "rephrased": "AI in healthcare: Game-changing potential! Machine learning enables early disease detection personalized medicine and predictive care at scale. Lives will be saved."
        },
        # Example 27
        {
            "draft": "Remote work is the future",
            "hook": "Remote work revolution! Distributed teams access global talent improve work-life balance and reduce costs. The future of work is flexible and global.",
            "concise": "Remote work is the future! Global talent better balance reduced costs. Flexibility wins.",
            "rephrased": "Remote work transformation! Companies embracing distributed teams access worldwide talent improve employee satisfaction and reduce operational costs. Future is flexible."
        },
        # Example 28
        {
            "draft": "Cloud computing reduces costs",
            "hook": "Cloud economics! Companies migrating to cloud reduce infrastructure costs 50 percent while increasing scalability. Pay for what you use transforms budgets.",
            "concise": "Cloud computing cuts costs! 50 percent infrastructure savings plus scalability. Smart economics.",
            "rephrased": "Cloud migration benefits! Organizations reduce infrastructure costs 50 percent while gaining elasticity and global reach. Pay-per-use model transforms economics."
        },
        # Example 29
        {
            "draft": "Cybersecurity threats are rising",
            "hook": "Cybersecurity alert! Attacks increased 300 percent in 2025. Organizations must prioritize security awareness training and zero-trust architecture. Protection is not optional.",
            "concise": "Cybersecurity threats rising! Attacks up 300 percent. Security must be priority.",
            "rephrased": "Cybersecurity landscape: Attacks surged 300 percent in 2025! Organizations need comprehensive security: awareness training zero-trust architecture continuous monitoring."
        },
        # Example 30
        {
            "draft": "DevOps improves deployment speed",
            "hook": "DevOps transformation! Teams using CI/CD deploy 200 times more frequently with 50 percent fewer failures. Automation and collaboration accelerate delivery.",
            "concise": "DevOps accelerates deployment! 200x more deploys 50 percent fewer failures. Automation works.",
            "rephrased": "DevOps impact! Organizations implementing CI/CD deploy 200 times more frequently with dramatically fewer failures. Automation and culture enable velocity."
        },
        # Example 31
        {
            "draft": "Completed data structures bootcamp",
            "hook": "Bootcamp complete! Mastered data structures and algorithms in 12 weeks. Intensive learning combined with daily practice builds deep expertise fast.",
            "concise": "Data structures bootcamp done! 12 weeks of intensive learning. Skills leveled up.",
            "rephrased": "Completed intensive data structures bootcamp! 12 weeks of focused learning and daily practice. Deep technical skills acquired through structured dedication."
        },
        # Example 32
        {
            "draft": "Built my portfolio website",
            "hook": "Portfolio live! Built personal website showcasing 15 projects using React and TypeScript. Online presence opens doors to opportunities.",
            "concise": "Portfolio website launched! 15 projects showcased. Online presence established.",
            "rephrased": "Personal portfolio launched! Website built with React and TypeScript showcasing 15 diverse projects. Strong online presence creates career opportunities."
        },
        # Example 33
        {
            "draft": "Got my first freelance client",
            "hook": "First client secured! Landed freelance web development project worth $5k. Side projects and networking create income opportunities beyond traditional employment.",
            "concise": "First freelance client! $5k web development project secured. Side income started.",
            "rephrased": "Freelance milestone! Secured first client for $5k web development project. Side projects and consistent networking create valuable opportunities."
        },
        # Example 34
        {
            "draft": "Mastered Docker and Kubernetes",
            "hook": "Container mastery! Learned Docker and Kubernetes enabling scalable deployments. Modern infrastructure skills are essential for cloud-native development.",
            "concise": "Docker and Kubernetes mastered! Scalable deployments enabled. Modern infrastructure learned.",
            "rephrased": "Container technology mastered! Docker and Kubernetes skills acquired enabling cloud-native deployments. Essential infrastructure knowledge for modern development."
        },
        # Example 35
        {
            "draft": "Led team of 10 developers",
            "hook": "Leadership milestone! Leading team of 10 engineers shipping features used by millions. Management is about enabling others to do their best work.",
            "concise": "Leading 10 developers! Shipping features for millions. Leadership responsibility accepted.",
            "rephrased": "Leadership role: Managing team of 10 talented engineers. Our work impacts millions of users. Leadership means enabling your team's best work."
        },
        # Example 36
        {
            "draft": "Created technical documentation system",
            "hook": "Documentation wins! Built knowledge base reducing onboarding time 70 percent. Good documentation is force multiplier for entire organization.",
            "concise": "Documentation system created! Onboarding time down 70 percent. Knowledge sharing improved.",
            "rephrased": "Documentation system launched! Comprehensive knowledge base reduces new hire onboarding 70 percent. Quality documentation multiplies organizational effectiveness."
        },
        # Example 37
        {
            "draft": "Implemented microservices architecture",
            "hook": "Architecture migration complete! Moved from monolith to microservices. Gained scalability and team autonomy but also complexity. Know your tradeoffs before committing.",
            "concise": "Microservices migration complete! Scalability and autonomy gained. Architecture evolved.",
            "rephrased": "Microservices architecture implemented! Migration from monolith complete. Gained scalability and team independence but added complexity. Understand tradeoffs first."
        },
        # Example 38
        {
            "draft": "Started weekly tech talks",
            "hook": "Tech talks launched! Weekly knowledge sharing sessions boost team learning and collaboration. Creating learning culture drives innovation.",
            "concise": "Weekly tech talks started! Team learning and collaboration boosted. Culture improved.",
            "rephrased": "Weekly tech talks initiated! Knowledge sharing sessions significantly improve team learning and collaboration. Investing in learning culture pays dividends."
        },
        # Example 39
        {
            "draft": "Built fraud detection system",
            "hook": "Fraud detection live! Machine learning system catching 98 percent of fraudulent transactions in real-time. AI protects customers and revenue.",
            "concise": "Fraud detection deployed! 98 percent catch rate in real-time. ML protection works.",
            "rephrased": "Fraud detection system launched! ML-powered solution catches 98 percent of fraudulent transactions in real-time. Artificial intelligence protects customers and business."
        },
        # Example 40
        {
            "draft": "Celebrating 3 years at Google",
            "hook": "Three years at Google! From nervous newcomer to confident contributor shipping features for billions. Collaborative culture pushes everyone to excel.",
            "concise": "3 years at Google! From newcomer to contributor. Growth continues.",
            "rephrased": "Google anniversary! Three years of growth from first day jitters to shipping features used by billions globally. Collaborative culture drives excellence."
        },
    ]
    
    # Optionally augment from AI Studio CSV-derived examples
    try:
        ai_studio_json = os.path.join('data', 'ai_studio_dataset.json')
        # If not created yet, try to generate via augment script
        if not os.path.exists(ai_studio_json):
            augment_script = os.path.join(os.path.dirname(__file__), 'augment_dataset_from_ai_studio.py')
            if os.path.exists(augment_script):
                os.system(f"python {augment_script}")
        if os.path.exists(ai_studio_json):
            with open(ai_studio_json, 'r', encoding='utf-8') as f:
                extra = json.load(f)
            # Normalize keys to our schema
            normalized = []
            for e in extra:
                normalized.append({
                    'draft': e.get('draft', ''),
                    'hook': e.get('hook') or e.get('engaging_hook') or e.get('story', ''),
                    'concise': e.get('concise') or e.get('concise_version') or '',
                    'rephrased': e.get('rephrased') or ''
                })
            data.extend([x for x in normalized if x['draft'] and x['hook'] and x['rephrased']])
            print(f"Augmented with AI Studio examples: +{len(normalized)}")
    except Exception as e:
        print(f"Warning: failed to augment from ai_studio_code.txt: {e}")

    # Shuffle the data for better training
    random.shuffle(data)
    
    # Split into training and test sets (80/20 split)
    split_point = int(0.8 * len(data))
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save datasets
    with open('data/train_data.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Training dataset saved: {len(train_data)} samples")
    
    with open('data/test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Test dataset saved: {len(test_data)} samples")
    
    with open('data/full_dataset.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Full dataset saved: {len(data)} samples")
    
    print("\n✅ Dataset preparation complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == "__main__":
    create_linkedin_dataset()
