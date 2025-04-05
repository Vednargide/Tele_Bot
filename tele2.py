import os
import logging
import re
import math
import asyncio
import google.generativeai as genai
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure API tokens
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Configure Gemini
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
        self.gemini_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }

    def clean_response(self, text):
        if not text:
            return "❌ I couldn't generate a response."
        
        text = re.sub(r'```(\w+)?\n(.*?)\n```', self.format_code_block, text, flags=re.DOTALL)
        text = re.sub(r'\$(.+?)\$', r'📐 \1', text)
        text = re.sub(r'^\s*[-*]\s(.+)$', r'• \1', text, flags=re.MULTILINE)
        text = re.sub(r'^(#+)\s(.+)$', self.format_header, text, flags=re.MULTILINE)
        
        return self.add_decorative_elements(text.strip())

    def format_code_block(self, match):
        language = match.group(1) or ''
        code = match.group(2)
        return f"💻 Code ({language}):\n┌──────────────────\n│ {code.replace('│', '|')}\n└──────────────────"

    def format_header(self, match):
        level = len(match.group(1))
        text = match.group(2)
        decorators = ['🔷', '🔶', '📌', '💠', '🔸', '🔹']
        return f"\n{decorators[min(level-1, len(decorators)-1)]} {text.upper()}\n"

    def add_decorative_elements(self, text):
        if "math" in text.lower() or any(char in text for char in "+-×÷="):
            text = "🧮 Mathematical Solution:\n" + text
        elif "code" in text.lower() or "programming" in text.lower():
            text = "👨‍💻 Programming Solution:\n" + text
        elif "aptitude" in text.lower():
            text = "🎯 Aptitude Solution:\n" + text
        else:
            text = "💡 Answer:\n" + text

        text = re.sub(r'(Important:|Note:|Remember:)(.*?)(?=\n\n|$)', 
                     r'📢 \1\n━━━━━━━━━━━━━━\2\n━━━━━━━━━━━━━━', 
                     text, flags=re.DOTALL)

        text = re.sub(r'Step (\d+):', r'📍 Step \1:', text)

        if "conclusion" in text.lower():
            text = re.sub(r'(conclusion:.*?)(?=\n|$)', 
                         r'🎯 Final \1', 
                         text, flags=re.IGNORECASE)

        return f"{'═' * 30}\n{text}\n{'═' * 30}"

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

    async def get_response(self, query):
        try:
            if re.match(r'^[\d+\-*/().\s]+$', query):
                result = self.math.solve(query)
                if result is not None:
                    return self.format_math_response(f"Result: {result}")

            apt_type = self.aptitude.detect_type(query)
            if apt_type:
                prompt = f"Solve this {apt_type} problem with detailed steps: {query}"
            else:
                prompt = query

            response = await self.get_gemini_response(prompt)
            
            if apt_type:
                return self.format_aptitude_response(response)
            return self.format_general_response(response)

        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return "❌ I encountered an error. Please try rephrasing your question."

    def format_math_response(self, response):
        return f"""
🧮 Mathematical Calculation
━━━━━━━━━━━━━━━━━━━━
📊 Expression: {response.split('Result:')[0]}
📝 Result: {response.split('Result:')[1]}
━━━━━━━━━━━━━━━━━━━━"""

    def format_aptitude_response(self, response):
        formatted = "🎯 Aptitude Problem Solution\n"
        formatted += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        steps = response.split('\n')
        for i, step in enumerate(steps, 1):
            if step.strip():
                formatted += f"📍 Step {i}: {step}\n"
        
        formatted += "\n✨ Final Answer: " + steps[-1]
        return formatted

    def format_general_response(self, response):
        sections = response.split('\n\n')
        formatted = ""
        
        for i, section in enumerate(sections):
            if i == 0:
                formatted += f"💡 {section}\n\n"
            else:
                formatted += f"📌 {section}\n\n"
        
        return formatted

bot = AIBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 Welcome! I can help you with:\n\n"
        "📊 Mathematics\n"
        "🧮 Aptitude Problems\n"
        "📝 General Questions\n"
        "💡 Technical Queries\n\n"
        "Just ask me anything!"
    )
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "I can help with:\n\n"
        "• Math calculations\n"
        "• Percentage problems\n"
        "• Profit/Loss calculations\n"
        "• Time and Distance\n"
        "• Sequences and Series\n"
        "• General knowledge\n"
        "• Programming questions\n\n"
        "Just type your question!"
    )
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
        await update.message.reply_text("Sorry, I encountered an error. Please try again.")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
        handle_message
    ))
    
    application.run_polling()

if __name__ == '__main__':
    main()
