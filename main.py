"""
SEO Article Generation Service
A complete agent-based backend system for generating SEO-optimized articles
"""

import os
import json
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
import openai
import google.generativeai as genai

# ============================================================================
# DATA MODELS
# ============================================================================

class SERPResult(BaseModel):
    """Individual SERP result"""
    rank: int
    url: str
    title: str
    snippet: str
    relevance_score: Optional[float] = None

class KeywordAnalysis(BaseModel):
    """Keyword analysis results"""
    primary_keywords: List[str] = Field(default_factory=list)
    secondary_keywords: List[str] = Field(default_factory=list)
    keyword_density: Dict[str, float] = Field(default_factory=dict)

class SEOMetadata(BaseModel):
    """SEO metadata"""
    title_tag: str
    meta_description: str
    canonical_url: Optional[str] = None
    
    @validator('title_tag')
    def validate_title_length(cls, v):
        if len(v) > 60:
            raise ValueError('Title tag should be under 60 characters')
        return v
    
    @validator('meta_description')
    def validate_description_length(cls, v):
        if len(v) > 160:
            raise ValueError('Meta description should be under 160 characters')
        return v

class ArticleOutline(BaseModel):
    """Article outline structure"""
    h1_title: str
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_word_count: int

class Article(BaseModel):
    """Complete article output"""
    content: str
    seo_metadata: SEOMetadata
    keyword_analysis: KeywordAnalysis
    internal_links: List[str] = Field(default_factory=list)
    external_references: List[Dict[str, str]] = Field(default_factory=list)
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    faq_section: Optional[List[Dict[str, str]]] = None
    seo_score: Optional[float] = None

class JobStatus(BaseModel):
    """Job tracking"""
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Article] = None
    error: Optional[str] = None

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    SERPAPI_KEY = "40cb3842691cc75ab923c79931c898e903609317987cfa339113ef327388f3e0"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_MODEL = "gemini"  # or "openai"
    TARGET_WORD_COUNT = 1500
    
# ============================================================================
# SERP DATA FETCHER
# ============================================================================

class SERPFetcher:
    """Handles SERP data retrieval"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def fetch(self, query: str, language: str = 'en', num_results: int = 10) -> List[SERPResult]:
        """Fetch top SERP results"""
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "hl": language,
            "api_key": self.api_key,
            "num": num_results
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('organic_results', [])[:num_results]:
                results.append(SERPResult(
                    rank=item.get('position', 0),
                    url=item.get('link', ''),
                    title=item.get('title', ''),
                    snippet=item.get('snippet', '')
                ))
            return results
        except Exception as e:
            print(f"Error fetching SERP data: {e}")
            return self._get_mock_serp_results(query)
    
    def _get_mock_serp_results(self, query: str) -> List[SERPResult]:
        """Fallback mock SERP results"""
        return [
            SERPResult(
                rank=i+1,
                url=f"https://example{i+1}.com/article",
                title=f"Top Article About {query} - Result {i+1}",
                snippet=f"This is a comprehensive guide about {query}..."
            ) for i in range(10)
        ]

# ============================================================================
# SERP ANALYZER
# ============================================================================

class SERPAnalyzer:
    """Analyzes SERP results using sentence transformers for reranking"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def analyze(self, query: str, serp_results: List[SERPResult]) -> Dict[str, Any]:
        """Analyze SERP results and extract insights"""
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Encode all snippets and titles
        texts = [f"{r.title} {r.snippet}" for r in serp_results]
        text_embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # Calculate similarity scores
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, text_embeddings)[0].tolist()
        
        # Update relevance scores
        for i, result in enumerate(serp_results):
            result.relevance_score = scores[i]
        
        # Sort by relevance
        serp_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        # Extract common topics
        common_topics = self._extract_common_topics(serp_results)
        keywords = self._extract_keywords(serp_results)
        
        return {
            "ranked_results": serp_results,
            "common_topics": common_topics,
            "keywords": keywords,
            "avg_relevance": sum(scores) / len(scores) if scores else 0
        }
    
    def _extract_common_topics(self, results: List[SERPResult]) -> List[str]:
        """Extract common topics from titles and snippets"""
        from collections import Counter
        import re
        
        all_text = " ".join([f"{r.title} {r.snippet}" for r in results])
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        
        # Remove common stop words
        stop_words = {'that', 'this', 'with', 'from', 'have', 'will', 'your', 'more'}
        words = [w for w in words if w not in stop_words]
        
        common = Counter(words).most_common(15)
        return [word for word, _ in common]
    
    def _extract_keywords(self, results: List[SERPResult]) -> Dict[str, int]:
        """Extract and count keywords"""
        from collections import Counter
        import re
        
        all_text = " ".join([f"{r.title} {r.snippet}" for r in results])
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        return dict(Counter(words).most_common(30))

# ============================================================================
# AI CONTENT GENERATOR
# ============================================================================

class AIContentGenerator:
    """Generates content using OpenAI or Gemini"""
    
    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        
        if provider == "openai":
            openai.api_key = Config.OPENAI_API_KEY
            self.model = "gpt-4"
        else:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_completion(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text completion"""
        try:
            if self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            else:
                response = self.model.generate_content(prompt)
                return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error: {str(e)}"

# ============================================================================
# ARTICLE GENERATOR AGENT
# ============================================================================

class ArticleGeneratorAgent:
    """Main agent orchestrating article generation"""
    
    def __init__(self, provider: str = "gemini"):
        self.serp_fetcher = SERPFetcher(Config.SERPAPI_KEY)
        self.serp_analyzer = SERPAnalyzer()
        self.content_generator = AIContentGenerator(provider)
    
    def generate_article(self, query: str, target_word_count: int, language: str) -> Article:
        """Complete article generation pipeline"""
        print(f"\n🚀 Starting article generation for: '{query}'")
        
        # Step 1: Fetch SERP data
        print("\n📊 Step 1: Fetching SERP data...")
        serp_results = self.serp_fetcher.fetch(query, language)
        print(f"✓ Fetched {len(serp_results)} SERP results")
        
        # Step 2: Analyze SERP
        print("\n🔍 Step 2: Analyzing SERP results...")
        analysis = self.serp_analyzer.analyze(query, serp_results)
        print(f"✓ Identified {len(analysis['common_topics'])} common topics")
        
        # Step 3: Generate outline
        print("\n📝 Step 3: Generating article outline...")
        outline = self._generate_outline(query, analysis, target_word_count)
        print(f"✓ Created outline with {len(outline.sections)} sections")
        
        # Step 4: Generate content
        print("\n✍️  Step 4: Generating article content...")
        content = self._generate_content(query, outline, analysis, target_word_count)
        print(f"✓ Generated {len(content.split())} words")
        
        # Step 5: Generate SEO metadata
        print("\n🏷️  Step 5: Creating SEO metadata...")
        seo_metadata = self._generate_seo_metadata(query, content)
        
        # Step 6: Generate keyword analysis
        keyword_analysis = self._analyze_keywords(content, analysis['keywords'])
        
        # Step 7: Generate suggestions
        print("\n🔗 Step 6: Generating links and references...")
        internal_links = self._suggest_internal_links(query, content)
        external_refs = self._suggest_external_references(analysis['ranked_results'][:4])
        
        # Step 8: Generate structured data
        structured_data = self._generate_structured_data(query, content, seo_metadata)
        
        # Step 9: Generate FAQ
        faq = self._generate_faq(query, analysis)
        
        # Calculate SEO score
        seo_score = self._calculate_seo_score(content, keyword_analysis, seo_metadata)
        
        print("\n✅ Article generation complete!")
        
        return Article(
            content=content,
            seo_metadata=seo_metadata,
            keyword_analysis=keyword_analysis,
            internal_links=internal_links,
            external_references=external_refs,
            structured_data=structured_data,
            faq_section=faq,
            seo_score=seo_score
        )
    
    def _generate_outline(self, query: str, analysis: Dict, word_count: int) -> ArticleOutline:
        """Generate article outline"""
        topics = ", ".join(analysis['common_topics'][:10])
        
        prompt = f"""Create an SEO-optimized article outline for: "{query}"

Based on top-ranking content, common topics include: {topics}

Requirements:
- Target word count: {word_count}
- Natural, engaging structure
- Include H2 and H3 headings
- Focus on user intent and value

Generate a JSON outline with this structure:
{{
  "h1_title": "Main title",
  "sections": [
    {{"h2": "Section title", "h3_subsections": ["Subsection 1", "Subsection 2"], "word_count": 300}}
  ]
}}"""
        
        response = self.content_generator.generate_completion(prompt, 1000)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                outline_data = json.loads(json_match.group())
                return ArticleOutline(
                    h1_title=outline_data.get('h1_title', query),
                    sections=outline_data.get('sections', []),
                    estimated_word_count=word_count
                )
        except:
            pass
        
        # Fallback outline
        return ArticleOutline(
            h1_title=f"Complete Guide to {query.title()}",
            sections=[
                {"h2": "Introduction", "word_count": 200},
                {"h2": "Key Concepts", "word_count": 400},
                {"h2": "Best Practices", "word_count": 400},
                {"h2": "Common Challenges", "word_count": 300},
                {"h2": "Conclusion", "word_count": 200}
            ],
            estimated_word_count=word_count
        )
    
    def _generate_content(self, query: str, outline: ArticleOutline, 
                         analysis: Dict, word_count: int) -> str:
        """Generate full article content"""
        sections_str = "\n".join([f"- {s.get('h2', s)}" for s in outline.sections])
        keywords = ", ".join(list(analysis['keywords'].keys())[:15])
        
        prompt = f"""Write a comprehensive, SEO-optimized article about: "{query}"

Article Title (H1): {outline.h1_title}

Sections to cover:
{sections_str}

Target length: {word_count} words
Important keywords: {keywords}

Requirements:
- Natural, engaging writing (not robotic)
- Use H1, H2, and H3 tags appropriately
- Include practical examples and insights
- Write for humans, not just search engines
- Start with a compelling introduction
- End with a strong conclusion
- Use transition words and varied sentence structure

Write the complete article in HTML format with proper heading tags."""
        
        content = self.content_generator.generate_completion(prompt, 3000)
        
        # Ensure HTML structure
        if not content.startswith('<'):
            content = f"<h1>{outline.h1_title}</h1>\n\n{content}"
        
        return content
    
    def _generate_seo_metadata(self, query: str, content: str) -> SEOMetadata:
        """Generate SEO metadata"""
        prompt = f"""Create SEO metadata for an article about: "{query}"

Content preview: {content[:500]}...

Generate:
1. Title tag (50-60 chars, compelling, includes keyword)
2. Meta description (150-160 chars, action-oriented)

Format as JSON:
{{"title_tag": "...", "meta_description": "..."}}"""
        
        response = self.content_generator.generate_completion(prompt, 300)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return SEOMetadata(
                    title_tag=data.get('title_tag', query)[:60],
                    meta_description=data.get('meta_description', '')[:160]
                )
        except:
            pass
        
        return SEOMetadata(
            title_tag=f"{query.title()} - Complete Guide"[:60],
            meta_description=f"Discover everything about {query}. Expert insights and practical tips."[:160]
        )
    
    def _analyze_keywords(self, content: str, serp_keywords: Dict) -> KeywordAnalysis:
        """Analyze keywords in content"""
        import re
        from collections import Counter
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        word_freq = Counter(words)
        total_words = len(words)
        
        # Identify primary keywords (high frequency + in SERP)
        primary = []
        for word in serp_keywords:
            if word in word_freq and word_freq[word] > 3:
                primary.append(word)
                if len(primary) >= 5:
                    break
        
        # Secondary keywords
        secondary = [w for w, c in word_freq.most_common(15) if w not in primary][:10]
        
        # Calculate density
        density = {w: (word_freq[w] / total_words) * 100 for w in primary}
        
        return KeywordAnalysis(
            primary_keywords=primary,
            secondary_keywords=secondary,
            keyword_density=density
        )
    
    def _suggest_internal_links(self, query: str, content: str) -> List[str]:
        """Suggest internal linking opportunities"""
        suggestions = [
            f"/blog/{query.lower().replace(' ', '-')}-guide",
            f"/resources/{query.lower().replace(' ', '-')}",
            f"/case-studies/{query.lower().replace(' ', '-')}-examples"
        ]
        return suggestions[:3]
    
    def _suggest_external_references(self, top_results: List[SERPResult]) -> List[Dict[str, str]]:
        """Suggest authoritative external references"""
        refs = []
        for result in top_results[:4]:
            if any(domain in result.url for domain in ['.edu', '.gov', '.org']):
                refs.append({
                    "title": result.title,
                    "url": result.url,
                    "description": result.snippet[:100]
                })
        
        # Add generic authorities if not enough
        while len(refs) < 2:
            refs.append({
                "title": "Industry Authority Source",
                "url": "https://example.com",
                "description": "Additional authoritative reference"
            })
        
        return refs[:4]
    
    def _generate_structured_data(self, query: str, content: str, 
                                  metadata: SEOMetadata) -> Dict[str, Any]:
        """Generate schema.org structured data"""
        return {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": metadata.title_tag,
            "description": metadata.meta_description,
            "author": {
                "@type": "Organization",
                "name": "Your Company"
            },
            "datePublished": datetime.now().isoformat(),
            "dateModified": datetime.now().isoformat()
        }
    
    def _generate_faq(self, query: str, analysis: Dict) -> List[Dict[str, str]]:
        """Generate FAQ section"""
        topics = ", ".join(analysis['common_topics'][:5])
        
        prompt = f"""Generate 3-5 frequently asked questions about: "{query}"

Related topics: {topics}

Format as JSON array:
[{{"question": "...", "answer": "..."}}]"""
        
        response = self.content_generator.generate_completion(prompt, 500)
        
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return [
            {"question": f"What is {query}?", "answer": f"A comprehensive overview of {query}."},
            {"question": f"Why is {query} important?", "answer": f"Understanding {query} is crucial for..."}
        ]
    
    def _calculate_seo_score(self, content: str, keywords: KeywordAnalysis, 
                            metadata: SEOMetadata) -> float:
        """Calculate overall SEO score"""
        score = 0.0
        
        # Word count (target 1500)
        word_count = len(content.split())
        if 1200 <= word_count <= 2000:
            score += 20
        elif word_count >= 800:
            score += 10
        
        # Keyword usage
        if len(keywords.primary_keywords) >= 3:
            score += 20
        
        # Metadata quality
        if len(metadata.title_tag) >= 40:
            score += 15
        if len(metadata.meta_description) >= 120:
            score += 15
        
        # Content structure (H1, H2, H3 tags)
        if '<h1>' in content.lower():
            score += 10
        if content.lower().count('<h2>') >= 3:
            score += 10
        if content.lower().count('<h3>') >= 2:
            score += 10
        
        return min(score, 100.0)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("   SEO ARTICLE GENERATOR")
    print("=" * 60)
    
    # Input
    query = input("\n📝 Enter topic/keyword: ").strip()
    if not query:
        query = "best productivity tools for remote teams"
    
    language = input("🌍 Language (English/French/Spanish) [English]: ").strip().capitalize()
    if language not in ["English", "French", "Spanish"]:
        language = "English"
    
    provider = input("🤖 AI Provider (openai/gemini) [gemini]: ").strip().lower()
    if provider not in ["openai", "gemini"]:
        provider = "gemini"
    
    target_words = Config.TARGET_WORD_COUNT
    
    print(f"\n📋 Configuration:")
    print(f"   Topic: {query}")
    print(f"   Word Count: {target_words}")
    print(f"   Language: {language}")
    print(f"   AI Provider: {provider.upper()}")
    
    # Generate article
    agent = ArticleGeneratorAgent(provider)
    article = agent.generate_article(query, target_words, language.lower()[:2])
    
    # Display results
    print("\n" + "=" * 60)
    print("   RESULTS")
    print("=" * 60)
    
    print(f"\n📊 SEO Score: {article.seo_score:.1f}/100")
    print(f"\n🏷️  Title Tag: {article.seo_metadata.title_tag}")
    print(f"📝 Meta Description: {article.seo_metadata.meta_description}")
    print(f"\n🔑 Primary Keywords: {', '.join(article.keyword_analysis.primary_keywords[:5])}")
    print(f"\n🔗 Internal Links: {len(article.internal_links)}")
    print(f"📚 External References: {len(article.external_references)}")
    print(f"\n❓ FAQ Items: {len(article.faq_section or [])}")
    
    # Save to file
    output_file = f"article_{query.replace(' ', '_')[:30]}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(article.dict(), f, indent=2, default=str)
    
    print(f"\n💾 Full article saved to: {output_file}")
    print("\n✅ Generation complete!")

if __name__ == "__main__":
    main()
