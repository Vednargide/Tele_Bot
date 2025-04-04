import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import json
from bs4 import BeautifulSoup
import re
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

class EnhancedAIBot:
    def __init__(self):
        self.math_pattern = re.compile(r'[\d\+\-\*\/\(\)\s]+$')
    
    def evaluate_mathematical_expression(self, expression):
        """Safely evaluate mathematical expressions"""
        try:
            clean_expr = expression.replace(' ', '')
            if not self.math_pattern.match(clean_expr):
                return None
            return eval(clean_expr, {"__builtins__": {}}, {"math": math})
        except:
            return None

    async def get_answer(self, query):
        """Get answer from various sources"""
        try:
            # Check if it's a math question
            if self.math_pattern.match(query):
                result = self.evaluate_mathematical_expression(query)
                if result is not None:
                    return f"The answer is: {result}"

            # Use DuckDuckGo API for general knowledge
            encoded_query = requests.utils.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
            response = requests.get(url)
            data = response.json()

            if data.get("Abstract"):
                return data["Abstract"]
            elif data.get("Answer"):
                return data["Answer"]

            # Fallback to dictionary API for definitions
            if query.lower().startswith("what is"):
                term = query.lower().replace("what is", "").replace("?", "").strip()
                dict_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
                dict_response = requests.get(dict_url)
                
                if dict_response.status_code == 200:
                    dict_data = dict_response.json()
                    if dict_data and len(dict_data) > 0:
                        meaning = dict_data[0].get('meanings', [{}])[0]
                        definition = meaning.get('definitions', [{}])[0].get('definition', '')
                        if definition:
                            return f"Definition: {definition}"

            return "I apologize, but I need more context to provide an accurate answer to this question."

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "I encountered an error processing your question. Please try rephrasing it."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "👋 Welcome! I'm an AI assistant designed to provide accurate answers "
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
    
    # Get answer
    response = await ai_bot.get_answer(question)
    
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
