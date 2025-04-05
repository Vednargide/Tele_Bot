import os
import logging
import re
import asyncio
import google.generativeai as genai
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

class MathHandler:
    def solve(self, expression):
        try:
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            return eval(safe_expr, {"__builtins__": {}}, {})
        except:
            return None

class AptitudeHandler:
    def __init__(self):
        self.patterns = {
            'percentage': r'(\d+(\.\d+)?%|\bpercent\b)',
            'profit_loss': r'\b(profit|loss)\b',
            'time_distance': r'\b(speed|time|distance)\b',
            'ratio': r'\b(ratio|proportion)\b',
            'average': r'\b(average|mean)\b',
            'sequence': r'\b(sequence|series|next number)\b'
        }

    def detect_type(self, question):
        for qtype, pattern in self.patterns.items():
            if re.search(pattern, question.lower()):
                return qtype
        return None

class AIBot:
    def __init__(self):
        self.aptitude = AptitudeHandler()
        self.math = MathHandler()
        self.allowed_group_ids = [-1001369278049]
        self.response_cache = {}
        self.gemini_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        self.domain_patterns = {
            'programming': r'code|program|function|algorithm',
            'mathematics': r'math|calculate|solve|equation',
            'science': r'physics|chemistry|biology',
            'general': r'explain|what|how|why'
        }

    async def should_respond(self, chat_id, message_text):
        if not message_text or message_text.startswith('/'):
            return False
        return chat_id in self.allowed_group_ids

    async def get_gemini_response(self, prompt):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    def detect_domain(self, query):
        for domain, pattern in self.domain_patterns.items():
            if re.search(pattern, query.lower()):
                return domain
        return 'general'

    def format_response(self, text, domain):
        if not text:
            return "❌ I couldn't generate a response."

        formatted = "═══════════════════════════\n"
        
        if domain == 'programming':
            formatted += "👨‍💻 Programming Solution:\n\n"
        elif domain == 'mathematics':
            formatted += "🧮 Mathematical Solution:\n\n"
        elif domain == 'science':
            formatted += "🔬 Scientific Explanation:\n\n"
        else:
            formatted += "💡 Answer:\n\n"

        sections = text.split('\n')
        for section in sections:
            if section.strip():
                if 'step' in section.lower():
                    formatted += f"📍 {section}\n"
                elif ':' in section:
                    formatted += f"• {section}\n"
                else:
                    formatted += f"{section}\n"

        formatted += "\n═══════════════════════════"
        return formatted

    async def get_response(self, query):
        try:
            if query in self.response_cache:
                return self.response_cache[query]

            if re.match(r'^[\d+\-*/().\s]+$', query):
                result = self.math.solve(query)
                if result is not None:
                    return f"🧮 Result: {result}"

            domain = self.detect_domain(query)
            apt_type = self.aptitude.detect_type(query)

            if apt_type:
                prompt = f"Solve this {apt_type} problem step by step: {query}"
            else:
                prompt = f"Provide a detailed answer for: {query}"

            response = await self.get_gemini_response(prompt)
            formatted_response = self.format_response(response, domain)
            
            self.response_cache[query] = formatted_response
            return formatted_response

        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return "❌ I encountered an error. Please try again."

bot = AIBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """
🌟 Welcome! I can help you with:

📊 Mathematics
🧮 Aptitude Problems
📝 General Questions
💡 Technical Queries

Just ask me anything!
"""
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
I can help with:

• Math calculations
• Percentage problems
• Profit/Loss calculations
• Time and Distance
• Sequences and Series
• General knowledge
• Programming questions

Just type your question!
"""
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        message_text = update.message.text

        if not await bot.should_respond(chat_id, message_text):
            return

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        response = await bot.get_response(message_text)
        
        if len(response) <= 4096:
            await update.message.reply_text(response)
        else:
            chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk)
                
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error. Please try again.")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
        handle_message
    ))
    
    print("🤖 Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()
