import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from huggingface_hub import InferenceClient
import requests
import json
import asyncio
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Configure AI models
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

class AIBot:
    def __init__(self):
        self.gemini_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=2048,
        )
        
        self.hf_models = {
            'general': "meta-llama/Llama-2-70b-chat-hf",
            'code': "bigcode/starcoder2-15b",
            'math': "google/flan-t5-xxl"
        }

    async def get_response(self, query):
        try:
            # Get responses from both models
            gemini_response = await self.get_gemini_response(query)
            hf_response = await self.get_huggingface_response(query)
            
            # Combine responses
            final_response = await self.combine_responses(gemini_response, hf_response)
            return final_response
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return "I apologize, but I encountered an error processing your request."

    async def get_gemini_response(self, query):
        try:
            response = gemini_model.generate_content(
                query,
                generation_config=self.gemini_config
            )
            return response.text
        except:
            return None

    async def get_huggingface_response(self, query):
        try:
            response = hf_client.text_generation(
                prompt=query,
                model=self.hf_models['general'],
                max_new_tokens=512
            )
            return response
        except:
            return None

    async def combine_responses(self, gemini_resp, hf_resp):
        if gemini_resp and hf_resp:
            return f"Combined Analysis:\n\n{gemini_resp}\n\nAdditional Insights:\n{hf_resp}"
        return gemini_resp or hf_resp or "No response available."

bot = AIBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "👋 Welcome to AI Assistant!\n\n"
        "I can help you with:\n"
        "• Programming questions\n"
        "• Mathematical problems\n"
        "• General knowledge\n"
        "• And much more!\n\n"
        "Just ask me anything!"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "*Available Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "*Just ask any question and I'll help you!*"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    response = await bot.get_response(update.message.text)
    
    if len(response) <= 4096:
        await update.message.reply_text(response, parse_mode="Markdown")
    else:
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(
                f"Part {i+1}/{len(chunks)}:\n\n{chunk}",
                parse_mode="Markdown"
            )

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.run_polling()

if __name__ == "__main__":
    main()
