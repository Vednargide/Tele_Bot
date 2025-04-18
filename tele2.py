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
from PIL import Image
import easyocr
import cv2
import numpy as np
import io

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])  # EasyOCR for free OCR functionality

    async def process_image(self, image_bytes):
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to CV2 format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Enhance image for better OCR
            enhanced = self.enhance_image(cv_image)
            
            # Extract text using EasyOCR
            easy_result = self.reader.readtext(enhanced)
            if easy_result:
                text = ' '.join([t[1] for t in easy_result])
                return self.clean_text(text)
            
            return "Could not extract text from the image."
        
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return None

    def enhance_image(self, image):
        try:
            # Grayscale conversion
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return sharpened
        except Exception as e:
            logger.error(f"Image enhancement error: {str(e)}")
            return image

    def clean_text(self, text):
        cleaned = re.sub(r'[^\w\s\n.?!]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()


class AIBot:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.allowed_group_ids = [-1001369278049]

        # Models optimized for CPU
        self.qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2', device=-1)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.classifier = pipeline('text-classification', model='facebook/bart-large-mnli', device=-1)

        self.confidence_threshold = 0.80

    async def should_respond(self, chat_id, message_text):
        return chat_id in self.allowed_group_ids and message_text.strip()

    async def process_image_query(self, image_bytes):
        try:
            extracted_text = await self.image_processor.process_image(image_bytes)
            if not extracted_text:
                return "❌ Could not process the image. Please try a clearer image."

            prompt = f"Analyze this text and provide a detailed response:\n\n{extracted_text}"
            response = await self.get_gemini_response(prompt)
            return f"📝 Extracted Text:\n{extracted_text}\n\n💡 Response:\n{response}"
        except Exception as e:
            logger.error(f"Image query error: {str(e)}")
            return "❌ Error processing image query."

    async def get_gemini_response(self, prompt):
        try:
            response = gemini_model.generate_content(prompt)
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'parts') and response.parts:
                return ' '.join(part.text for part in response.parts)
            return "I couldn't process this request."
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return "❌ Error generating response."

    async def get_response(self, query):
        try:
            response = await self.get_gemini_response(query)
            return self.clean_response(response)
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return "❌ Error processing your request."

    def clean_response(self, text):
        text = text.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
        text = re.sub(r'\n{3,}', '\n\n', text)
        return "💡 " + text.strip()


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
    await update.message.reply_text("🛑 Saale Kamine So jaa... Goodbye!")
    await context.application.stop()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id

        if update.message.photo:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            photo = update.message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            photo_bytes = await photo_file.download_as_bytearray()
            response = await bot.process_image_query(photo_bytes)
        else:
            message_text = update.message.text
            if not await bot.should_respond(chat_id, message_text):
                return
            response = await bot.get_response(message_text)

        if len(response) <= 4096:
            await update.message.reply_text(response)
        else:
            for chunk in (response[i:i+4000] for i in range(0, len(response), 4000)):
                await update.message.reply_text(chunk)
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error. Please try again.")


def main():
    load_dotenv()
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=GEMINI_API_KEY)
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_message))

    print("🤖 Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
