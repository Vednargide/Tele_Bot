import yt_dlp
 import asyncio
 import os
 import logging
 from yt_dlp import YoutubeDL
 from telegram import Update
 from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
 import logging
 import os
 BOT_TOKEN = os.getenv('BOT_TOKEN')
 ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID')
 
 # Replace with your actual bot token
 
 BOT_TOKEN = os.getenv("BOT_TOKEN")
 ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID"))
 
 
 # Set up logging
 @@ -30,48 +34,18 @@ async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
         "Add 'bgm' to get only the background music (e.g., 'song_name bgm')."
     )
 
 async def download_song(song_name: str) -> str:
     """Downloads a regular song and returns the file path."""
     search_query = song_name
     output_file = f"{song_name}.%(ext)s"
 
     ydl_opts = {
         'format': 'bestaudio/best',
         'outtmpl': output_file,
         'noplaylist': True,
     }
 
     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
         try:
             info = ydl.extract_info(f"ytsearch:{search_query}", download=True)
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
 
 async def download_bgm(song_name: str) -> str:
     """Downloads the BGM version of a song and returns the file path."""
     search_query = f"{song_name} background music"
     output_file = f"{song_name}_bgm.%(ext)s"
 
 async def download_song(song_name: str, user_id: int) -> str:
     """Downloads a song or BGM and returns the file path."""
     output_file = f"{user_id}_{song_name}.%(ext)s"
     ydl_opts = {
         'format': 'bestaudio/best',
         'outtmpl': output_file,
         'noplaylist': True,
     }
 
     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
     with YoutubeDL(ydl_opts) as ydl:
         try:
             info = ydl.extract_info(f"ytsearch:{search_query}", download=True)
             info = ydl.extract_info(f"ytsearch:{song_name}", download=True)
             file_path = ydl.prepare_filename(info)
 
             if os.path.exists(file_path):
 @@ -87,8 +61,8 @@ async def download_bgm(song_name: str) -> str:
             raise FileNotFoundError(f"An error occurred during download: {e}")
 
 async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
     """Handles song requests and replies with user details."""
     song_name = update.message.text
     """Handles song requests from users."""
     song_name = update.message.text.strip()
     user = update.message.from_user
     chat_id = update.effective_chat.id
 
 @@ -98,50 +72,45 @@ async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
 
     try:
         if "bgm" in song_name.lower():
             # Handle BGM request
             song_name = song_name.lower().replace("bgm", "").strip()
             await update.message.reply_text(f"Searching for BGM of '{song_name}'...")
             file_path = await download_bgm(song_name)
             print(f"DEBUG: Sending BGM file from {file_path} to user {chat_id} ({user.full_name})")
             bgm_file_path = await download_song(f"{song_name} background music", user.id)
 
             with open(file_path, 'rb') as audio_file:
             # Send BGM
             with open(bgm_file_path, 'rb') as audio_file:
                 await context.bot.send_audio(
                     chat_id=chat_id,
                     audio=audio_file,
                     caption=f"Here is your BGM: {song_name}"
                 )
 
             os.remove(file_path)
             print(f"DEBUG: Deleted BGM file {file_path}")
             os.remove(bgm_file_path)
         else:
             # Handle both song and BGM
             await update.message.reply_text(f"Searching for both song and BGM of '{song_name}'...")
             
             # Download song and BGM concurrently
             song_task = download_song(song_name, user.id)
             bgm_task = download_song(f"{song_name} background music", user.id)
             song_file_path, bgm_file_path = await asyncio.gather(song_task, bgm_task)
 
             # Download the regular song
             song_file_path = await download_song(song_name)
             print(f"DEBUG: Sending song file from {song_file_path} to user {chat_id} ({user.full_name})")
 
             # Send song
             with open(song_file_path, 'rb') as audio_file:
                 await context.bot.send_audio(
                     chat_id=chat_id,
                     audio=audio_file,
                     caption=f"Here is your song: {song_name}"
                 )
 
             os.remove(song_file_path)
             print(f"DEBUG: Deleted song file {song_file_path}")
 
             # Download the BGM version
             bgm_file_path = await download_bgm(song_name)
             print(f"DEBUG: Sending BGM file from {bgm_file_path} to user {chat_id} ({user.full_name})")
 
             # Send BGM
             with open(bgm_file_path, 'rb') as audio_file:
                 await context.bot.send_audio(
                     chat_id=chat_id,
                     audio=audio_file,
                     caption=f"Here is your BGM: {song_name}"
                 )
 
             os.remove(bgm_file_path)
             print(f"DEBUG: Deleted BGM file {bgm_file_path}")
 
     except FileNotFoundError as e:
         await update.message.reply_text(f"Error: {e}")
