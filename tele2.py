import os
import logging
import re
import asyncio
import google.generativeai as genai
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = los.getenv("GEMINI_API_KEY")
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

class AIBot:
    def __init__(self):
        self.aptitude = AptitudeHandler()
        self.math = MathHandler()
        self.pattern_recognition = PatternRecognitionHandler()
        self.allowed_group_ids = [-1001369278049]
        self.gemini_config = {
            'temperature': 0.3,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 4096,
        }

    async def should_respond(self, chat_id, message_text):
        if not message_text or message_text.startswith('/'):
            return False
        return chat_id in self.allowed_group_ids

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
        
            text = text.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
            text = re.sub(r'\n{3,}', '\n\n', text)
        
            return "💡 " + text.strip()

    async def get_enhanced_response(self, query):
        try:
            responses = []
            
            # Get Gemini response
            gemini_response = await self.get_gemini_response(query)
            if gemini_response:
                responses.append(("Gemini", gemini_response, 0.9))

            # Get specialized responses based on query type
            query_type = self.classify_query(query)
            
            if query_type == "question":
                qa_response = self.qa_model(question=query, context=query)
                responses.append(("QA", qa_response['answer'], qa_response['score']))
            
            # Calculate response quality scores
            best_response = self.select_best_response(query, responses)
            
            if best_response:
                return f"🎯 [{best_response[0]}] {best_response[1]}"
            
            return await self.get_gemini_response(query)  # Fallback to Gemini

        except Exception as e:
            logger.error(f"Enhanced response error: {str(e)}")
            return await self.get_gemini_response(query)

    def classify_query(self, query):
        result = self.classifier(query, candidate_labels=["question", "statement", "command"])
        return result[0]['label']

    def select_best_response(self, query, responses):
        if not responses:
            return None

        best_score = 0
        best_response = None

        query_embedding = self.text_model.encode(query, convert_to_tensor=True)

        for source, response, conf in responses:
            if conf < self.confidence_threshold:
                continue

            response_embedding = self.text_model.encode(response, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, response_embedding)
            
            score = float(similarity[0][0]) * conf
            
            if score > best_score:
                best_score = score
                best_response = (source, response)

        return best_response

    async def get_response(self, query):
        try:
            # Check for math first
            if re.match(r'^[\d+\-*/().\s]+$', query):
                result = self.math.solve(query)
                if result is not None:
                    return f"🔢 Result: {result}"

            # Get enhanced response
            response = await self.get_enhanced_response(query)
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
    """Handles the /stop command to stop the bot."""
    await update.message.reply_text("🛑 Saale Kamine So jaa... Goodbye!")
    await context.application.stop()

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
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
        handle_message
    ))
    
    print("🤖 Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
