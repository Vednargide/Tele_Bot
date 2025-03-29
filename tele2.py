import yt_dlp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os

# Replace 'YOUR_BOT_TOKEN' with your actual bot token


BOT_TOKEN = os.getenv("BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    await update.message.reply_text("Hi! Send me the name of a song, and I'll fetch it for you!")

import os

async def download_song_without_ffmpeg(song_name: str, bgm: bool = False) -> str:
    """Downloads the song or BGM version and returns the file path."""
    output_file = f"{song_name}{'_bgm' if bgm else ''}.%(ext)s"  # Template for output filename
    ydl_opts = {
        'format': 'bestaudio/best',  # Download the best audio format available
        'outtmpl': output_file,  # Output filename template
        'noplaylist': True,  # Ensure only a single result is downloaded
    }

    # Adjust audio quality for BGM if needed
    if bgm:
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',  # Lower quality for BGM
        }]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Download the audio and retrieve file info
            info = ydl.extract_info(f"ytsearch:{song_name}", download=True)
            file_path = ydl.prepare_filename(info)  # Initial file path
            print(file_path+"fhfhfhfhf")

            # Check if the file exists and handle cases where extension is missing or incorrect
            if not os.path.exists(file_path):
                print(f"DEBUG: Initial file path {file_path} not found. Checking alternatives...")

                # Look for common extensions
                base_path = file_path.rsplit('.', 1)[0]
                possible_extensions = ['.mp4']
                for ext in possible_extensions:
                    
                    test_path = base_path + ext
                    print(test_path+"hfhfhfhf")
                    if os.path.exists(test_path):
                        print(f"DEBUG: Found file with extension {ext}")
                        return test_path
                    
                print(test_path+"fhfhfhfhf")

                # If no valid file is found, raise an error
                raise FileNotFoundError(f"Downloaded file not found for {song_name}")

            return file_path

        except Exception as e:
            raise FileNotFoundError(f"An error occurred during download: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles song requests and replies with user details."""
    song_name = update.message.text  # The song name provided by the user
    chat_id = update.effective_chat.id  # Chat ID to send the file
    user = update.message.from_user  # Get user info

    # Retrieve the first name and last name (if available)
    first_name = user.first_name if user.first_name else "Unknown"
    last_name = user.last_name if user.last_name else ""
    full_name = f"{first_name} {last_name}".strip()

    try:
        # Inform the user that the bot is searching for the song
        await update.message.reply_text(f"Searching for '{song_name}'...")

        # Download the song
        file_path = await download_song_without_ffmpeg(song_name)

        # Log the file path for debugging
        print(f"DEBUG: Sending file from {file_path} to user {chat_id} ({full_name})")

        # Send the file to the user
        with open(file_path, 'rb') as audio_file:
            await context.bot.send_audio(
                chat_id=chat_id,
                audio=audio_file,
                caption=f"Here is your song: {song_name}"
            )


        # Optionally delete the file after sending
        os.remove(file_path)
        print(f"DEBUG: Deleted file {file_path}")

    except FileNotFoundError as fnf_error: 
        file_path
        
    


def main():
    """Main function to start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()  