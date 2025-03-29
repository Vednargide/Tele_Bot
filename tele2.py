import yt_dlp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os

# Replace with your actual bot token
BOT_TOKEN = os.getenv("BOT_TOKEN")



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    await update.message.reply_text(
        "Hi! Send me the name of a song (add 'BGM' if you only want the background music)."
    )


async def download_song_without_ffmpeg(song_name: str, bgm: bool = False) -> str:
    """Downloads the song or BGM version and returns the file path."""
    output_file = f"{song_name}{'_bgm' if bgm else ''}.mp4"  # Direct download to .mp4
    ydl_opts = {
        'format': 'bestaudio',  # Download the best audio-only format
        'outtmpl': output_file,  # Output filename template
        'noplaylist': True,  # Ensure only a single result is downloaded
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Download the audio
            info = ydl.extract_info(f"ytsearch:{song_name}", download=True)
            print(f"DEBUG: File downloaded as {output_file}")
            return output_file

        except Exception as e:
            raise FileNotFoundError(f"An error occurred during download: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles song requests and replies with user details."""
    song_name = update.message.text  # The song name provided by the user
    chat_id = update.effective_chat.id  # Chat ID to send the file
    user = update.message.from_user  # Get user info

    # Determine if the user specifically requested a BGM version
    is_bgm_only = "bgm" in song_name.lower()

    # If BGM is specifically requested, clean the song name
    clean_song_name = song_name.replace("BGM", "").replace("bgm", "").strip()

    try:
        # Inform the user that the bot is searching for the song/BGM
        if is_bgm_only:
            await update.message.reply_text(f"Searching for the BGM of '{clean_song_name}'...")
        else:
            await update.message.reply_text(f"Searching for '{clean_song_name}' and its BGM...")

        # Handle BGM only
        if is_bgm_only:
            bgm_file = await download_song_without_ffmpeg(clean_song_name, bgm=True)
            print(f"DEBUG: Sending BGM file: {bgm_file}")

            with open(bgm_file, 'rb') as bgm_audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=bgm_audio_file,
                    caption=f"Here is the BGM for: {clean_song_name}"
                )

            os.remove(bgm_file)
            print(f"DEBUG: Deleted BGM file: {bgm_file}")

        # Handle both song and BGM
        else:
            song_file = await download_song_without_ffmpeg(clean_song_name)
            bgm_file = await download_song_without_ffmpeg(clean_song_name, bgm=True)

            print(f"DEBUG: Sending files to user {chat_id}: {song_file}, {bgm_file}")

            with open(song_file, 'rb') as audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    caption=f"Here is the song: {clean_song_name}"
                )

            with open(bgm_file, 'rb') as bgm_audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=bgm_audio_file,
                    caption=f"Here is the BGM for: {clean_song_name}"
                )

            os.remove(song_file)
            os.remove(bgm_file)
            print(f"DEBUG: Deleted files: {song_file} and {bgm_file}")

    except FileNotFoundError as error:
        await update.message.reply_text(
            f"Sorry, I couldn't find the song or its BGM for '{clean_song_name}'. Please try again."
        )
        print(f"ERROR: {error}")


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
