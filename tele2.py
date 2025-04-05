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
 
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
 
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)
 
class MathHandler:
     def solve(self, expression):
        try:
             # Remove any unsafe operations
             safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
             return eval(safe_expr, {"__builtins__": {}}, {})
         except:
             return None
 
class AptitudeHandler:
    def __init__(self):
         self.math_handler = MathHandler()
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
 
    def format_solution(self, steps, answer):
         return f"Step-by-step Solution:\n{steps}\n\nFinal Answer: {answer}"
 
class AIBot:
     def __init__(self):
         self.aptitude = AptitudeHandler()
         self.math = MathHandler()
         self.allowed_group_ids = [-1001369278049]  # Replace with your group ID
         self.gemini_config = {
             'temperature': 0.7,
             'top_p': 0.9,
             'top_k': 40,
             'max_output_tokens': 2048,
         }
     
 
     
     async def should_respond(self, chat_id, message_text):
         # Skip empty messages or messages starting with '/'
         if not message_text or message_text.startswith('/'):
             return False
             
         # Check if message is from allowed group
         return chat_id in self.allowed_group_ids
     async def get_response(self, query):
         try:
             # Check for simple math
             if re.match(r'^[\d+\-*/().\s]+$', query):
                 result = self.math.solve(query)
                 if result is not None:
                     return f"Result: {result}"
 
             # Check for aptitude question
             apt_type = self.aptitude.detect_type(query)
             if apt_type:
                 prompt = f"Solve this {apt_type} problem with detailed steps: {query}"
             else:
                 prompt = query
 
             # Get Gemini response
             response = await self.get_gemini_response(prompt)
 
             # Format and clean response
             return self.clean_response(response)
 
         except Exception as e:
             logger.error(f"Error in get_response: {e}")
             return "I encountered an error. Please try rephrasing your question."
 
     async def get_gemini_response(self, prompt):
         try:
             response = gemini_model.generate_content(prompt)
             return response.text
         except Exception as e:
             logger.error(f"Gemini API error: {e}")
             return None
 
     def clean_response(self, text):
         if not text:
             return "I couldn't generate a response."
 
         # Clean markdown characters
         text = text.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
 
         # Remove multiple newlines
         text = re.sub(r'\n{3,}', '\n\n', text)
 
         return text.strip()
 
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
         "• Math calculations (e.g., 2+2)\n"
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
         await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
         response = await bot.get_response(update.message.text)
         # Get chat ID and message text
         chat_id = update.effective_chat.id
         message_text = update.message.text
 
         # Check if bot should respond
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
 
     # Add handlers
     application.add_handler(CommandHandler("start", start))
     application.add_handler(CommandHandler("help", help_command))
     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
     application.add_handler(MessageHandler(
         filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
         handle_message
     ))
 
     application.run_polling()
 
 if __name__ == "__main__":
     main()
