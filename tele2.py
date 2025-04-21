import os
import logging
import re
import asyncio
import google.generativeai as genai
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,  # Add this import
    filters, 
    ContextTypes
)

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

class PatternRecognitionHandler:
    def __init__(self):
        self.transformation_types = {
            'letter_shift': self._check_letter_shift,
            'rearrangement': self._check_rearrangement,
            'position_swap': self._check_position_swap,
            'letter_replacement': self._check_replacement
        }

    def analyze_pattern(self, question):
        pairs = re.findall(r'([A-Z]+)\s*:\s*([A-Z]+)', question)
        if not pairs:
            return None

        analysis = []
        for source, target in pairs:
            patterns = []
            for pattern_type, checker in self.transformation_types.items():
                if result := checker(source, target):
                    patterns.append((pattern_type, result))
            analysis.append((source, target, patterns))

        return self._format_pattern_analysis(analysis)

    def _check_letter_shift(self, source, target):
        if len(source) != len(target):
            return None
        shifts = []
        for s, t in zip(source, target):
            shift = (ord(t) - ord(s)) % 26
            shifts.append(shift)
        return shifts if len(set(shifts)) <= 2 else None

    def _check_rearrangement(self, source, target):
        return sorted(source) == sorted(target)

    def _check_position_swap(self, source, target):
        if len(source) != len(target):
            return None
        swaps = []
        for i, (s, t) in enumerate(zip(source, target)):
            if s != t:
                swaps.append((i, target.index(s)))
        return swaps if swaps else None

    def _check_replacement(self, source, target):
        if len(source) != len(target):
            return None
        replacements = {}
        for s, t in zip(source, target):
            if s != t:
                replacements[s] = t
        return replacements if replacements else None

    def _format_pattern_analysis(self, analysis):
        response = "🔍 Pattern Analysis:\n═══════════════════\n\n"
        
        for source, target, patterns in analysis:
            response += f"📌 {source} ➡️ {target}:\n"
            for pattern_type, details in patterns:
                if pattern_type == 'letter_shift':
                    response += "   • Letter shifting pattern detected\n"
                    response += f"     Shifts: {details}\n"
                elif pattern_type == 'rearrangement':
                    response += "   • Letters are rearranged\n"
                elif pattern_type == 'position_swap':
                    response += "   • Position swapping detected\n"
                    response += f"     Swaps: {details}\n"
                elif pattern_type == 'letter_replacement':
                    response += "   • Letter replacement pattern\n"
                    response += f"     Replacements: {details}\n"
            response += "\n"

        similar_pairs = self._find_similar_patterns(analysis)
        if similar_pairs:
            response += "✨ Similar Transformations Found:\n"
            for pair in similar_pairs:
                response += f"• {pair[0]} and {pair[1]} share the same pattern\n"
        else:
            response += "❗ No two pairs share exactly the same transformation pattern\n"

        return response

    def _find_similar_patterns(self, analysis):
        similar = []
        for i in range(len(analysis)):
            for j in range(i + 1, len(analysis)):
                if self._compare_patterns(analysis[i][2], analysis[j][2]):
                    similar.append((analysis[i][0], analysis[j][0]))
        return similar

    def _compare_patterns(self, pattern1, pattern2):
        if len(pattern1) != len(pattern2):
            return False
        return all(p1[0] == p2[0] and p1[1] == p2[1] for p1, p2 in zip(pattern1, pattern2))

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

# Add these imports at the top
import io
from PIL import Image
import pytesseract

# After existing imports
# Configure pytesseract path (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AIBot:
    def __init__(self):
        self.aptitude = AptitudeHandler()
        self.math = MathHandler()
        self.pattern_recognition = PatternRecognitionHandler()
        self.allowed_group_ids = [-1001369278049]
        self.programming_questions = {}
        self.gemini_config = {
            'temperature': 0.3,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 4096,
        }
        self.is_active = True
        
    async def should_respond(self, chat_id, message_text):
        if not message_text or message_text.startswith('/'):
            return False
        return chat_id in self.allowed_group_ids and self.is_active  # Modify this line

    async def get_gemini_response(self, prompt):
        try:
            response = gemini_model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
            elif response and hasattr(response, 'parts'):
                return ' '.join(part.text for part in response.parts)
            else:
                return "I couldn't process this request properly."
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return "I encountered an error processing your request."
    def clean_response(self, text):
            if not text:
                return "❌ I couldn't generate a response."
        
            
        
            return "💡 " + text.strip()

    def _is_programming_question(self, text):
        keywords = [
            'program', 'code', 'function', 'algorithm',
            'write a', 'implement', 'create a program', 'Constraints:',
            'Input:', 'Output:', 'Example', 'return'  # Add these keywords
        ]
        return any(keyword.lower() in text.lower() for keyword in keywords)

    async def get_response(self, query, chat_id=None):
        try:
            # Check if it's a programming question
            if chat_id and self._is_programming_question(query):
                # Store the question
                self.programming_questions[chat_id] = query
                # Create language selection buttons
                keyboard = [[
                    InlineKeyboardButton("🐍 Python", callback_data="lang_python"),
                    InlineKeyboardButton("☕ Java", callback_data="lang_java"),
                ], [
                    InlineKeyboardButton("⚡ C++", callback_data="lang_cpp"),
                    InlineKeyboardButton("💛 JavaScript", callback_data="lang_javascript")
                ]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                return ("Please select the programming language:", reply_markup)

        # Check for simple math
            if re.match(r'^[\d+\-*/().\s]+$', query):
                result = self.math.solve(query)
                if result is not None:
                    return f"🔢 Result: {result}"

        # Get Gemini response with error handling
            response = await self.get_gemini_response(query)
            if not response:
                return "❌ I couldn't generate a response. Please try again."
            
            return self.clean_response(response)

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return "❌ I encountered an error. Please try rephrasing your question."


    
        try:
        # Remove any problematic characters
            text = str(text).replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
        # Remove excessive newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
        # Add emoji prefix
            return "💡 " + text.strip()
        except Exception as e:
            logger.error(f"Error in clean_response: {str(e)}")
            return "❌ Error formatting response"
        

bot = AIBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """
🌟 Welcome! I can help you with:

📊 Mathematics
🧮 Aptitude Problems
🔍 Pattern Recognition
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
• Pattern Recognition
• Sequences and Series
• General knowledge
• Programming questions

Just type your question!
"""
    await update.message.reply_text(help_text)
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /stop command to temporarily stop the bot."""
    bot.is_active = False  # Just deactivate the bot
    await update.message.reply_text("🛑 Bot is now sleeping! Use /start to wake me up.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot.is_active = True  # Reactivate the bot
    welcome_text = """
🌟 Welcome! I can help you with:

📊 Mathematics
🧮 Aptitude Problems
🔍 Pattern Recognition
📝 General Questions
💡 Technical Queries

Just ask me anything!
"""
    await update.message.reply_text(welcome_text)

# Add new callback handler
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        
        if query.data.startswith("lang_"):
            language = query.data.split("_")[1]
            chat_id = query.message.chat_id
            
            if chat_id in bot.programming_questions:
                question = bot.programming_questions[chat_id]
                prompt = f"""Write a solution in {language} for the following problem:
                
{question}

Provide:
1. Problem analysis
2. Solution approach
3. Complete code with comments
4. Example usage
"""
                response = await bot.get_gemini_response(prompt)
                await query.edit_message_text(text=bot.clean_response(response))
                del bot.programming_questions[chat_id]  # Clean up
            else:
                await query.edit_message_text(text="❌ Question not found. Please ask again.")
                
    except Exception as e:
        logger.error(f"Error in button callback: {e}")
        await query.edit_message_text(text="❌ Error processing selection. Please try again.")

# Modify handle_message function
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        message_text = update.message.text

        if not await bot.should_respond(chat_id, message_text):
            return

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        response = await bot.get_response(message_text, chat_id)
        
        # Check if response is a tuple (for programming questions)
        if isinstance(response, tuple):
            message_text, reply_markup = response
            await update.message.reply_text(message_text, reply_markup=reply_markup)
        else:
            if len(response) <= 4096:
                await update.message.reply_text(response)
            else:
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
                
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error. Please try again.")

# Update main() function to work with Railway
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
        handle_message
    ))
    # Add photo handler
    application.add_handler(MessageHandler(
        filters.PHOTO & filters.ChatType.GROUPS,
        handle_photo
    ))
    
    print("🤖 Bot is starting...")
    
    # Use webhook for Railway deployment
    PORT = int(os.environ.get('PORT', 8080))
    
    # Check if running on Railway
    if os.environ.get('RAILWAY_STATIC_URL'):
        # Use webhook when on Railway
        WEBHOOK_URL = os.environ.get('RAILWAY_STATIC_URL')
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            webhook_url=WEBHOOK_URL
        )
    else:
        # Use polling for local development
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
