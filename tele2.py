
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import json
import wolframalpha
from transformers import pipeline
from dotenv import load_dotenv
import re
import math

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

class EnhancedAIBot:
    def __init__(self):
        # Initialize free models and tools
        self.qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')
        self.math_pattern = re.compile(r'[\d\+\-\*\/\(\)\s]+$')
    
    def evaluate_mathematical_expression(self, expression):
        """Safely evaluate mathematical expressions"""
        try:
            # Remove any whitespace and validate expression
            clean_expr = expression.replace(' ', '')
            if not self.math_pattern.match(clean_expr):
                return None
            
            return eval(clean_expr, {"__builtins__": {}}, {"math": math})
        except:
            return None

    async def get_wikipedia_context(self, query):
        """Get relevant context from Wikipedia"""
        try:
            url = f"https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "utf8": 1
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if "query" in data and "search" in data["query"]:
                return data["query"]["search"][0]["snippet"]
            return ""
        except:
            return ""

    async def process_question(self, question):
        """Process questions with enhanced accuracy"""
        try:
            # Check if it's a mathematical question
            if self.math_pattern.match(question):
                result = self.evaluate_mathematical_expression(question)
                if result is not None:
                    return f"The answer is: {result}"

            # Get context from Wikipedia
            context = await self.get_wikipedia_context(question)
            
            # Use QA model for natural language questions
            if context:
                answer = self.qa_pipeline(question=question, context=context)
                confidence = answer['score']
                
                if confidence > 0.8:
                    return f"Answer: {answer['answer']}\n(Confidence: {confidence:.2%})"

            # Fallback to rule-based responses for common questions
            if "what is" in question.lower():
                # Process definitional questions
                definition = await self.get_definition(question)
                if definition:
                    return definition

            # Generate response using combination of available tools
            response = await self.generate_combined_response(question)
            return response

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "I apologize, but I encountered an error processing your question. Please try rephrasing it."

    async def get_definition(self, question):
        """Get definitions for 'what is' questions"""
        try:
            term = question.lower().replace("what is", "").replace("?", "").strip()
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    meanings = data[0].get('meanings', [])
                    if meanings:
                        definition = meanings[0].get('definitions', [{}])[0].get('definition', '')
                        if definition:
                            return f"Definition: {definition}"
            return None
        except:
            return None

    async def generate_combined_response(self, question):
        """Generate response using multiple sources"""
        responses = []
        
        # Get Wikipedia context
        wiki_context = await self.get_wikipedia_context(question)
        if wiki_context:
            responses.append(wiki_context)

        # Use QA pipeline
        if responses:
            answer = self.qa_pipeline(question=question, context=" ".join(responses))
            if answer['score'] > 0.7:
                return f"Based on available information: {answer['answer']}"

        return "I apologize, but I don't have enough information to provide an accurate answer to this question."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "👋 Welcome! I'm an enhanced AI assistant designed to provide accurate answers "
        "to your questions. I can handle:\n\n"
        "• Mathematical calculations\n"
        "• General knowledge questions\n"
        "• Definitions and explanations\n\n"
        "Just ask me anything!"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "*Available Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "*Examples of questions you can ask:*\n"
        "• Mathematical: '2 + 2'\n"
        "• General: 'What is photosynthesis?'\n"
        "• Definitions: 'What is artificial intelligence?'"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

ai_bot = EnhancedAIBot()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages"""
    question = update.message.text
    
    # Send typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Process the question
    response = await ai_bot.process_question(question)
    
    # Send response
    await update.message.reply_text(response, parse_mode="Markdown")

def main():
    """Start the bot"""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
