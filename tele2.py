import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import json
from bs4 import BeautifulSoup
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get Telegram token from environment variable
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Initialize executor for parallel processing
executor = ThreadPoolExecutor(max_workers=5)

class AptitudeProgrammingBot:
    """Class to handle specialized responses for aptitude, programming, and MCQs"""
    
    @staticmethod
    async def search_stackoverflow(query):
        """Search Stack Overflow for programming questions"""
        try:
            search_url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={query}&site=stackoverflow"
            response = requests.get(search_url)
            data = response.json()
            
            results = []
            for item in data.get('items', [])[:5]:
                results.append({
                    "title": item.get('title'),
                    "link": item.get('link'),
                    "score": item.get('score'),
                    "answer_count": item.get('answer_count'),
                    "is_answered": item.get('is_answered')
                })
            
            return results
        except Exception as e:
            logger.error(f"Stack Overflow API error: {e}")
            return []

    @staticmethod
    async def search_github(query):
        """Search GitHub for code examples"""
        try:
            search_url = f"https://api.github.com/search/code?q={query}"
            headers = {"Accept": "application/vnd.github.v3+json"}
            response = requests.get(search_url, headers=headers)
            data = response.json()
            
            results = []
            for item in data.get('items', [])[:5]:
                repo = item.get('repository', {})
                results.append({
                    "name": item.get('name'),
                    "path": item.get('path'),
                    "repository": repo.get('full_name'),
                    "url": item.get('html_url')
                })
            
            return results
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
            return []

    @staticmethod
    async def search_web(query):
        """Perform a web search to get information"""
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            search_results = []
            for result in soup.select('.tF2Cxc'):
                title = result.select_one('.DKV0Md').text if result.select_one('.DKV0Md') else ""
                link = result.select_one('.yuRUbf a')['href'] if result.select_one('.yuRUbf a') else ""
                snippet = result.select_one('.VwiC3b').text if result.select_one('.VwiC3b') else ""
                if title and link and snippet:
                    search_results.append({"title": title, "link": link, "snippet": snippet})
            
            return search_results[:5]
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    @staticmethod
    async def search_wikipedia(query):
        """Get information from Wikipedia API"""
        try:
            # Search for Wikipedia articles
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            search_response = requests.get(search_url)
            search_data = search_response.json()
            
            if not search_data.get('query', {}).get('search', []):
                return None
            
            # Get the first result
            first_result = search_data['query']['search'][0]
            page_id = first_result['pageid']
            
            # Get the content of the article
            content_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&pageids={page_id}&format=json"
            content_response = requests.get(content_url)
            content_data = content_response.json()
            
            extract = content_data['query']['pages'][str(page_id)]['extract']
            title = content_data['query']['pages'][str(page_id)]['title']
            
            return {
                "title": title,
                "extract": extract,
                "url": f"https://en.wikipedia.org/?curid={page_id}"
            }
        except Exception as e:
            logger.error(f"Wikipedia API error: {e}")
            return None

    @staticmethod
    def detect_question_type(query):
        """Detect the type of question (aptitude, programming, MCQ)"""
        # Programming patterns
        programming_patterns = [
            r'code', r'program', r'function', r'algorithm', r'syntax',
            r'java', r'python', r'c\+\+', r'javascript', r'html', r'css',
            r'sql', r'database', r'api', r'error', r'debug', r'compile',
            r'runtime', r'exception', r'class', r'object', r'method',
            r'variable', r'array', r'list', r'dictionary', r'set', r'loop',
            r'if.*else', r'switch', r'case', r'recursion', r'iteration'
        ]
        
        # Aptitude patterns
        aptitude_patterns = [
            r'math', r'problem', r'solve', r'equation', r'calculate',
            r'percentage', r'ratio', r'proportion', r'average', r'probability',
            r'permutation', r'combination', r'profit', r'loss', r'interest',
            r'time.*work', r'distance', r'speed', r'time', r'algebra',
            r'geometry', r'trigonometry', r'arithmetic', r'progression',
            r'series', r'sequence', r'logical reasoning', r'puzzle'
        ]
        
        # MCQ patterns
        mcq_patterns = [
            r'options', r'choose', r'correct answer', r'multiple choice',
            r'mcq', r'quiz', r'option [abcd]', r'option [1234]',
            r'[abcd]\)', r'[1234]\)', r'which of the following'
        ]
        
        # Check for programming patterns
        for pattern in programming_patterns:
            if re.search(pattern, query.lower()):
                return "programming"
        
        # Check for aptitude patterns
        for pattern in aptitude_patterns:
            if re.search(pattern, query.lower()):
                return "aptitude"
        
        # Check for MCQ patterns
        for pattern in mcq_patterns:
            if re.search(pattern, query.lower()):
                return "mcq"
        
        # Default to general
        return "general"

    @classmethod
    async def generate_programming_response(cls, query):
        """Generate response for programming questions"""
        # Search Stack Overflow and GitHub in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, lambda: asyncio.run(cls.search_stackoverflow(query))),
            loop.run_in_executor(executor, lambda: asyncio.run(cls.search_github(query))),
            loop.run_in_executor(executor, lambda: asyncio.run(cls.search_web(f"programming {query} solution example")))
        ]
        
        stackoverflow_results, github_results, web_results = await asyncio.gather(*tasks)
        
        # Construct response
        response_parts = []
        
        response_parts.append(f"🖥️ *Programming Question: {query}*\n")
        
        # Add Stack Overflow results
        if stackoverflow_results:
            response_parts.append("📚 *Stack Overflow Solutions:*")
            for i, result in enumerate(stackoverflow_results[:3], 1):
                answered_status = "✅ Answered" if result['is_answered'] else "❓ Not Answered"
                response_parts.append(f"{i}. [{result['title']}]({result['link']})\n   Score: {result['score']} | {answered_status} | Answers: {result['answer_count']}\n")
        
        # Add GitHub code examples
        if github_results:
            response_parts.append("💻 *GitHub Code Examples:*")
            for i, result in enumerate(github_results[:3], 1):
                response_parts.append(f"{i}. [{result['name']} in {result['repository']}]({result['url']})\n   Path: {result['path']}\n")
        
        # Add web results
        if web_results:
            response_parts.append("🔍 *Additional Resources:*")
            for i, result in enumerate(web_results[:2], 1):
                response_parts.append(f"{i}. [{result['title']}]({result['link']})\n   {result['snippet'][:150]}...\n")
        
        # Add tips based on common programming topics
        if "error" in query.lower() or "exception" in query.lower():
            response_parts.append("💡 *Debugging Tips:*\n1. Check for syntax errors\n2. Print variable values\n3. Use a debugger\n4. Read the error message carefully\n")
        elif "algorithm" in query.lower():
            response_parts.append("💡 *Algorithm Tips:*\n1. Break down the problem\n2. Consider time and space complexity\n3. Look for existing algorithms that solve similar problems\n")
        
        return "\n".join(response_parts)

    @classmethod
    async def generate_aptitude_response(cls, query):
        """Generate response for aptitude questions"""
        # Search web and Wikipedia in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, lambda: asyncio.run(cls.search_web(f"aptitude {query} solution"))),
            loop.run_in_executor(executor, lambda: asyncio.run(cls.search_wikipedia(query)))
        ]
        
        web_results, wiki_info = await asyncio.gather(*tasks)
        
        # Construct response
        response_parts = []
        
        response_parts.append(f"🧮 *Aptitude Question: {query}*\n")
        
        # Add Wikipedia information if available
        if wiki_info:
            response_parts.append(f"📚 *Background Information:*\n{wiki_info['extract'][:300]}...\n[Read more]({wiki_info['url']})\n")
        
        # Add web results
        if web_results:
            response_parts.append("🔍 *Solutions and Explanations:*")
            for i, result in enumerate(web_results[:4], 1):
                response_parts.append(f"{i}. [{result['title']}]({result['link']})\n   {result['snippet'][:150]}...\n")
        
        # Add general approach for solving aptitude problems
        if "percentage" in query.lower():
            response_parts.append("💡 *Percentage Formula:*\nPercentage = (Value/Total Value) × 100\n")
        elif "profit" in query.lower() or "loss" in query.lower():
            response_parts.append("💡 *Profit & Loss Formulas:*\nProfit = SP - CP\nLoss = CP - SP\nProfit% = (Profit/CP) × 100\nLoss% = (Loss/CP) × 100\n")
        elif "interest" in query.lower():
            response_parts.append("💡 *Interest Formulas:*\nSimple Interest (SI) = (P × R × T)/100\nCompound Interest (CI) = P(1 + R/100)^T - P\nWhere P = Principal, R = Rate, T = Time\n")
        elif "time" in query.lower() and "work" in query.lower():
            response_parts.append("💡 *Time and Work Formula:*\nIf A can do a work in x days and B can do the same work in y days, then together they can finish the work in (xy)/(x+y) days\n")
        elif "speed" in query.lower() or "distance" in query.lower() or "time" in query.lower():
            response_parts.append("💡 *Speed, Distance & Time Formula:*\nSpeed = Distance/Time\nDistance = Speed × Time\nTime = Distance/Speed\n")
        
        return "\n".join(response_parts)

    @classmethod
    async def generate_mcq_response(cls, query):
        """Generate response for MCQ questions"""
        # Extract options if present
        options_match = re.search(r'(?:options|choices)(?:\s*:|\s+are)?(.+?)(?:\.|$)', query, re.DOTALL | re.IGNORECASE)
        options = []
        
        if options_match:
            options_text = options_match.group(1)
            # Try to extract options in different formats
            option_patterns = [
                r'([A-D])[\.:\)](.+?)(?=(?:[A-D][\.:\)]|$))',  # A) option text
                r'(\d)[\.:\)](.+?)(?=(?:\d[\.:\)]|$))',        # 1) option text
                r'([a-d])[\.\s](.+?)(?=(?:[a-d][\.\s]|$))'     # a. option text
            ]
            
            for pattern in option_patterns:
                found_options = re.findall(pattern, options_text, re.IGNORECASE)
                if found_options:
                    options = [(label.strip(), text.strip()) for label, text in found_options]
                    break
        
        # Search web for the question
        web_results = await cls.search_web(query)
        
        # Construct response
        response_parts = []
        
        response_parts.append(f"📝 *MCQ Question: {query}*\n")
        
        # Display extracted options if found
        if options:
            response_parts.append("*Options:*")
            for label, text in options:
                response_parts.append(f"  {label}) {text}")
            response_parts.append("")
        
        # Add web results
        if web_results:
            response_parts.append("🔍 *Relevant Information:*")
            for i, result in enumerate(web_results[:3], 1):
                response_parts.append(f"{i}. [{result['title']}]({result['link']})\n   {result['snippet'][:150]}...\n")
        
        # Add analysis based on the question and options
        response_parts.append("🧠 *Analysis:*")
        
        # Try to determine the correct answer based on web results
        likely_answer = None
        confidence = "Low"
        
        if web_results and options:
            # Simple algorithm to find the most mentioned option in search results
            option_mentions = {label: 0 for label, _ in options}
            option_texts = {text.lower(): label for label, text in options}
            
            for result in web_results:
                snippet = result['snippet'].lower()
                title = result['title'].lower()
                
                # Check for direct mentions of options
                for label, text in options:
                    text_lower = text.lower()
                    if text_lower in snippet or text_lower in title:
                        option_mentions[label] += 1
                        
                        # Check for phrases indicating correctness
                        correctness_phrases = ['correct answer', 'right answer', 'answer is', 'solution is']
                        for phrase in correctness_phrases:
                            if phrase in snippet and text_lower in snippet[snippet.find(phrase):snippet.find(phrase) + 100]:
                                option_mentions[label] += 2
            
            # Find the option with the most mentions
            if option_mentions:
                max_mentions = max(option_mentions.values())
                if max_mentions > 0:
                    likely_answers = [label for label, mentions in option_mentions.items() if mentions == max_mentions]
                    likely_answer = likely_answers[0]
                    confidence = "Medium" if max_mentions > 2 else "Low"
        
        if likely_answer:
            response_parts.append(f"Based on the information gathered, the most likely answer is *Option {likely_answer}*. (Confidence: {confidence})")
            response_parts.append(f"This is based on frequency of mentions and context in search results.")
        else:
            response_parts.append("Based on the available information, I couldn't determine a definitive answer with confidence.")
            response_parts.append("Please review the provided resources to make an informed decision.")
        
        response_parts.append("\n⚠️ *Note:* This analysis is based on web search results and may not be 100% accurate. Always verify the answer.")
        
        return "\n".join(response_parts)

    @classmethod
    async def generate_general_response(cls, query):
        """Generate response for general questions"""
        # Search web and Wikipedia in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, lambda: asyncio.run(cls.search_web(query))),
            loop.run_in_executor(executor, lambda: asyncio.run(cls.search_wikipedia(query)))
        ]
        
        web_results, wiki_info = await asyncio.gather(*tasks)
        
        # Construct response
        response_parts = []
        
        response_parts.append(f"❓ *Question: {query}*\n")
        
        # Add Wikipedia information if available
        if wiki_info:
            response_parts.append(f"📚 *From Wikipedia ({wiki_info['title']}):*\n{wiki_info['extract'][:300]}...\n[Read more]({wiki_info['url']})\n")
        
        # Add web results
        if web_results:
            response_parts.append("🔍 *Relevant Information:*")
            for i, result in enumerate(web_results[:4], 1):
                response_parts.append(f"{i}. [{result['title']}]({result['link']})\n   {result['snippet'][:150]}...\n")
        
        return "\n".join(response_parts)

    @classmethod
    async def generate_enhanced_response(cls, query):
        """Generate an enhanced response based on question type"""
        # Detect question type
        question_type = cls.detect_question_type(query)
        
        # Generate response based on question type
        if question_type == "programming":
            return await cls.generate_programming_response(query)
        elif question_type == "aptitude":
            return await cls.generate_aptitude_response(query)
        elif question_type == "mcq":
            return await cls.generate_mcq_response(query)
        else:
            return await cls.generate_general_response(query)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm your specialized aptitude, programming, and MCQ assistant. Ask me any question, and I'll provide the most accurate answer possible!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
    I'm a specialized assistant for aptitude, programming, and MCQ questions.
    
    Commands:
    /start - Start the bot
    /help - Show this help message
    
    Question types I can help with:
    - Programming questions (code, algorithms, debugging)
    - Aptitude questions (math, reasoning, puzzles)
    - Multiple Choice Questions (with options)
    - General knowledge questions
    
    Just send me any question, and I'll do my best to provide the most accurate answer!
    """
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages and generate enhanced responses."""
    query = update.message.text
    
    # Send typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Generate enhanced response
    response = await AptitudeProgrammingBot.generate_enhanced_response(query)
    
    # Split response if it's too long for Telegram
    if len(response) <= 4096:
        await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)
    else:
        # Split into chunks of 4000 characters (with some buffer)
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(f"Part {i+1}/{len(chunks)}:\n\n{chunk}", 
                                           parse_mode="Markdown", 
                                           disable_web_page_preview=True)

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main()
