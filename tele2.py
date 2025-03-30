import asyncio
import os
import logging
from yt_dlp import YoutubeDL
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Replace with your actual bot token

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID"))


# Set up logging
logging.basicConfig(filename="bot_usage.log", level=logging.INFO, format="%(asctime)s - %(message)s")

async def log_activity(user, song_name):
    """Log user activity to a file."""
    logging.info(f"User: {user.full_name} (ID: {user.id}, Username: {user.username}), Requested: {song_name}")

async def notify_admin(context, user, song_name):
    """Notify admin about the user's activity."""
    message = (
        f"New Song Request:\n"
        f"User: {user.first_name} {user.last_name or ''} (ID: {user.id}, Username: @{user.username or 'N/A'})\n"
        f"Requested: {song_name}"
    )
    await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    await update.message.reply_text(
        "Hi! Send me the name of a song, and I'll fetch it for you!\n\n"
        "Add 'bgm' to get only the background music (e.g., 'song_name bgm')."
    )

async def download_song(song_name: str, user_id: int) -> str:
    """Downloads a song or BGM and returns the file path."""
    output_file = f"{user_id}_{song_name}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'noplaylist': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch:{song_name}", download=True)
            file_path = ydl.prepare_filename(info)

            if os.path.exists(file_path):
                return file_path

            base_path = file_path.rsplit('.', 1)[0]
            for ext in ['.mp4']:
                test_path = base_path + ext
                if os.path.exists(test_path):
                    return test_path

        except Exception as e:
            raise FileNotFoundError(f"An error occurred during download: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles song requests from users."""
    song_name = update.message.text.strip()
    user = update.message.from_user
    chat_id = update.effective_chat.id

    # Log and notify admin
    await log_activity(user, song_name)
    await notify_admin(context, user, song_name)

    try:
        if "bgm" in song_name.lower():
            # Handle BGM request
            song_name = song_name.lower().replace("bgm", "").strip()
            await update.message.reply_text(f"Searching for BGM of '{song_name}'...")
            bgm_file_path = await download_song(f"{song_name} background music", user.id)

            # Send BGM
            with open(bgm_file_path, 'rb') as audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    caption=f"Here is your BGM: {song_name}"
                )
            os.remove(bgm_file_path)
        else:
            # Handle both song and BGM
            await update.message.reply_text(f"Searching for both song and BGM of '{song_name}'...")
            
            # Download song and BGM concurrently
            song_task = download_song(song_name, user.id)
            bgm_task = download_song(f"{song_name} background music", user.id)
            song_file_path, bgm_file_path = await asyncio.gather(song_task, bgm_task)

            # Send song
            with open(song_file_path, 'rb') as audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    caption=f"Here is your song: {song_name}"
                )
            os.remove(song_file_path)

            # Send BGM
            with open(bgm_file_path, 'rb') as audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    caption=f"Here is your BGM: {song_name}"
                )
            os.remove(bgm_file_path)

    except FileNotFoundError as e:
        await update.message.reply_text(f"Error: {e}")

def main():
    """Main function to start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == '__main__':
    main()
