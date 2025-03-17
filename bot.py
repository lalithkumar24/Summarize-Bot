import os
import requests
import nltk
nltk.data.path.append(r"C:\\Users\\lalit\\OneDrive\\Desktop\\telegram\\nltk_data")
import logging
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import re
import sqlite3
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
from transformers import pipeline
from telethon import TelegramClient, events, types
from telethon.tl.types import PeerChannel, PeerChat, PeerUser, MessageEntityUrl
from telethon.tl.functions.messages import GetHistoryRequest
from dotenv import load_dotenv
from googletrans import Translator


load_dotenv()
translator = Translator()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print("NLTK data found successfully!")
except LookupError as e:
    print(f"Error: NLTK data not found: {e}")
    print("Please run the download_nltk.py script first.")
    exit(1)

API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPEN_SERV_API_KEY = os.getenv("OPEN_SERV_API_KEY")
OPEN_SERV_API_URL = os.getenv("OPEN_SERV_API_URL")

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    logger.info("Loaded transformer summarization model")
except Exception as e:
    logger.error(f"Error loading transformer model: {e}")
    logger.info("Falling back to extractive summarization only")
    summarizer = None

DB_FILE = "telegram_assistant.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
       
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY,
            title TEXT,
            type TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            message_id INTEGER,
            sender_id INTEGER,
            sender_name TEXT,
            timestamp TIMESTAMP,
            text TEXT,
            has_media BOOLEAN,
            FOREIGN KEY (chat_id) REFERENCES chats (chat_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracked_keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            keyword TEXT,
            added_by INTEGER,
            added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (chat_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            message_id INTEGER,
            url TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (chat_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER DEFAULT 0,
            question TEXT,
            answer TEXT,
            added_by INTEGER,
            added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (chat_id)
        )
        ''')
        
       
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages (chat_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracked_keywords_chat_id ON tracked_keywords (chat_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_urls_chat_id ON urls (chat_id)')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
        conn.close()
client = TelegramClient('telegram_ai_assistant', API_ID, API_HASH)

chat_messages = defaultdict(list)
tracked_keywords = defaultdict(set)
user_preferences = defaultdict(dict)

faqs = {
    "how to use this bot": "Send /help to get a list of commands. You can use /summarize to get a summary of recent messages.",
    "what can this bot do": "This bot can summarize conversations, extract important information, and answer frequently asked questions.",
    "who created this bot": "This bot was created as a project to help manage information overload in Telegram chats."
}
def send_to_openserv(message):
    headers = {
        "Authorization": f"Bearer {OPEN_SERV_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"query": message}
    try:
        response = requests.post(OPEN_SERV_API_URL, json=data, headers=headers)
        response.raise_for_status()  # Raises an error for HTTP issues
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenServ API error: {e}")
        return None


def save_message_to_db(chat_id, message):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
           
            cursor.execute("SELECT chat_id FROM chats WHERE chat_id = ?", (chat_id,))
            if not cursor.fetchone():
               
                cursor.execute(
                    "INSERT INTO chats (chat_id, title, type) VALUES (?, ?, ?)",
                    (chat_id, "Unknown", "group") 
                )
            
           
            sender_id = message.sender_id if hasattr(message, 'sender_id') else None
            sender_name = "Unknown"
            timestamp = message.date if hasattr(message, 'date') else datetime.now()
            text = message.text if hasattr(message, 'text') else ""
            has_media = hasattr(message, 'media') and message.media is not None
            message_id = message.id if hasattr(message, 'id') else None
            
            cursor.execute(
                "INSERT INTO messages (chat_id, message_id, sender_id, sender_name, timestamp, text, has_media) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_id, message_id, sender_id, sender_name, timestamp, text, has_media)
            )
            
           
           
           
            if hasattr(message, 'text') and message.text:
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                urls = re.findall(url_pattern, message.text)
                
                for url in urls:
                    cursor.execute(
                        "INSERT INTO urls (chat_id, message_id, url, timestamp) VALUES (?, ?, ?, ?)",
                        (chat_id, message_id, url, timestamp)
                    )
            
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving message to database: {e}")

def load_tracked_keywords():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT chat_id, keyword FROM tracked_keywords")
        for chat_id, keyword in cursor.fetchall():
            tracked_keywords[chat_id].add(keyword.lower())
        
        conn.close()
        logger.info(f"Loaded tracked keywords for {len(tracked_keywords)} chats")
    except Exception as e:
        logger.error(f"Error loading tracked keywords: {e}")

def load_faqs_from_db():
    global faqs
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT question, answer FROM faqs WHERE chat_id = 0")
        rows = cursor.fetchall()
        
        for question, answer in rows:
            faqs[question.lower()] = answer
        
        conn.close()
        logger.info(f"Loaded {len(rows)} FAQs from database")
    except Exception as e:
        logger.error(f"Error loading FAQs: {e}")

async def extract_urls(message):
    urls = []
    
   
    if hasattr(message, 'entities') and message.entities:
        for entity in message.entities:
            if isinstance(entity, MessageEntityUrl):
               
                start = entity.offset
                end = start + entity.length
                url = message.text[start:end]
                urls.append(url)
            elif hasattr(entity, 'url') and entity.url:
               
                urls.append(entity.url)
    
   
    if hasattr(message, 'text') and message.text:
       
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        regex_urls = re.findall(url_pattern, message.text)
        urls.extend(regex_urls)
    
   
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls

def preprocess_text(text):
   
    text = re.sub(r'http\S+', '', text)
   
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
   
    text = text.lower()
    return text

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stopwords]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stopwords]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in all_words:
            vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in all_words:
            vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
    
   
    for i in range(len(sentences)):
        row_sum = np.sum(similarity_matrix[i])
        if row_sum > 0:
            similarity_matrix[i] /= row_sum
    
    return similarity_matrix
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
   
    sent1 = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stopwords]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stopwords]
    
   
    all_words = list(set(sent1 + sent2))
    
   
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
   
    return 1 - cosine_distance(vector1, vector2)

def extractive_summarize(text, num_sentences=5):
    try:
        stop_words = stopwords.words('english')
        sentences = sent_tokenize(text)
        
       
        if len(sentences) <= num_sentences:
            return text
        
       
        similarity_matrix = build_similarity_matrix(sentences, stop_words)
        
       
        sentence_scores = np.sum(similarity_matrix, axis=1)
        
       
        ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-num_sentences:]]
        
       
        ranked_sentences.sort(key=lambda s: sentences.index(s))
        
        return ' '.join(ranked_sentences)
    except Exception as e:
        logger.error(f"Error in extractive summarization: {e}")
        return text 
def summarize_text(text, num_sentences=5):
    if not text or len(text.strip()) == 0:
        logger.debug("No text to summarize.")
        return "No text to summarize."
    
   
    if len(text.split()) < 50:
        logger.debug("Text is too short for summarization. Returning original text.")
        return text
    
   
    if summarizer and len(text.split()) > 100:
        try:
           
            truncated_text = ' '.join(text.split()[:1000])
            logger.debug("Using transformer summarization.")
            summary = summarizer(truncated_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            logger.error(f"Error with transformer summarization: {e}")
            logger.info("Falling back to extractive summarization.")
    
   
    logger.debug("Using extractive summarization.")
    return extractive_summarize(text, num_sentences)
def extract_keywords(text, num_keywords=5):
   
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
   
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
   
    word_freq = {}
    for word in filtered_words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
   
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:num_keywords]

def get_messages_by_sender(chat_id, sender_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT text FROM messages WHERE chat_id = ? AND sender_id = ? ORDER BY timestamp DESC",
            (chat_id, sender_id)
        )
        
        messages = [row[0] for row in cursor.fetchall() if row[0]]
        logger.info(f"Retrieved {len(messages)} messages for sender ID {sender_id} in chat {chat_id}")
        conn.close()
        return messages
    except Exception as e:
        logger.error(f"Error getting messages by sender: {e}")
        return []

def get_urls_from_db(chat_id, limit=10):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT url FROM urls WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit)
        )
        
        urls = [row[0] for row in cursor.fetchall()]
        conn.close()
        return urls
    except Exception as e:
        logger.error(f"Error getting URLs from database: {e}")
        return []

def get_messages_by_sender(chat_id, sender_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT text FROM messages WHERE chat_id = ? AND sender_id = ? ORDER BY timestamp DESC",
            (chat_id, sender_id)
        )
        
        messages = [row[0] for row in cursor.fetchall() if row[0]]
        logger.info(f"Retrieved {len(messages)} messages for sender ID {sender_id} in chat {chat_id}")
        conn.close()
        return messages
    except Exception as e:
        logger.error(f"Error getting messages by sender: {e}")
        return []
    
def get_messages_from_db(chat_id, limit=100):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT text FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit)
        )
        
        messages = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return messages
    except Exception as e:
        logger.error(f"Error getting messages from database: {e}")
        return []

async def generate_summary(chat_id, use_db=True):
   
    if chat_id in chat_messages and len(chat_messages[chat_id]) >= 5:
        message_texts = [msg.text for msg in chat_messages[chat_id][-100:] if hasattr(msg, 'text') and msg.text]
    elif use_db:
       
        message_texts = get_messages_from_db(chat_id, 100)
    else:
        return "No messages to summarize. Add me to a group chat and I'll start collecting messages."
    
   
    if len(message_texts) < 5:
        return (
            f"I've only collected {len(message_texts)} messages so far. "
            f"Let me collect more messages before generating a summary.\n\n"
            f"Here are the recent messages:\n\n" + 
            "\n".join([f"- {text[:50]}..." if len(text) > 50 else f"- {text}" for text in message_texts[:5]])
        )
    
    all_text = " ".join(message_texts)
    
   
    preprocessed_text = preprocess_text(all_text)
    
   
    summary = summarize_text(preprocessed_text)
    
   
    keywords = extract_keywords(preprocessed_text)
    keywords_str = ", ".join([f"{word} ({count})" for word, count in keywords])
    
   
    urls = get_urls_from_db(chat_id, 5)
    urls_str = "\n".join([f"• {url}" for url in urls]) if urls else "No recent URLs found"
    
   
    summary_report = (
        "📊 Chat Summary Report 📊\n\n"
        f"🔍 Key Points:\n{summary}\n\n"
        f"📈 Trending Keywords:\n{keywords_str}\n\n"
        f"🔗 Recent Links:\n{urls_str}"
    )
    
    return summary_report

async def generate_detailed_summary(chat_identifier=None, days=1, use_db=True):
    try:
        chat_id = None
        
       
        if chat_identifier:
            if isinstance(chat_identifier, str):
                chat = await get_chat_by_name(chat_identifier)
                if chat:
                    chat_id = chat.id
            else:
                chat_id = chat_identifier
        
        if not chat_id:
            return "Please provide a valid chat name or ID."
        
       
        since_date = datetime.now() - timedelta(days=days)
        
       
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
       
        cursor.execute("""
            SELECT m.text, m.timestamp, m.has_media, m.sender_name
            FROM messages m
            WHERE m.chat_id = ? AND m.timestamp > ?
            ORDER BY m.timestamp DESC
        """, (chat_id, since_date))
        
        messages = cursor.fetchall()
        
       
        cursor.execute("""
            SELECT url, timestamp
            FROM urls
            WHERE chat_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (chat_id, since_date))
        
        urls = cursor.fetchall()
        
        conn.close()
        
        if not messages:
            return f"No messages found in the specified timeframe."
        
       
        message_texts = [msg[0] for msg in messages if msg[0]]
        all_text = " ".join(message_texts)
        
       
        summary = summarize_text(all_text) if all_text else "No text to summarize."
        
       
        keywords = extract_keywords(all_text) if all_text else []
        keywords_str = ", ".join([f"{word} ({count})" for word, count in keywords])
        
       
        media_count = sum(1 for msg in messages if msg[2])
        
       
        formatted_urls = []
        for url, timestamp in urls:
            dt = datetime.fromisoformat(timestamp)
            formatted_urls.append(f"• {url} (Posted: {dt.strftime('%Y-%m-%d %H:%M')})")
        
       
        timeline = defaultdict(int)
        for msg in messages:
            dt = datetime.fromisoformat(msg[1])
            hour = dt.strftime('%Y-%m-%d %H:00')
            timeline[hour] += 1
        
       
        detailed_summary = (
            "📊 Detailed Chat Summary Report 📊\n\n"
            f"📅 Time Period: Last {days} day(s)\n"
            f"📝 Total Messages: {len(messages)}\n"
            f"📷 Media Messages: {media_count}\n"
            f"🔗 Shared URLs: {len(urls)}\n\n"
            
            "🔍 Key Points:\n"
            f"{summary}\n\n"
            
            "📈 Trending Keywords:\n"
            f"{keywords_str}\n\n"
            
            "⏰ Activity Timeline:\n" +
            "\n".join([f"• {hour}: {count} messages" for hour, count in sorted(timeline.items())]) +
            "\n\n"
            
            "🔗 Recent Links:\n" +
            ("\n".join(formatted_urls) if formatted_urls else "No URLs shared") +
            "\n"
        )
        
        return detailed_summary
        
    except Exception as e:
        logger.error(f"Error generating detailed summary: {e}")
        return "Error generating summary. Please try again later."

async def process_message_content(event):
    try:
        chat_id = event.chat_id
        message_text = event.message.text.lower() if event.message.text else ""
        
       
        if chat_id in tracked_keywords:
            for keyword in tracked_keywords[chat_id]:
                if keyword in message_text:
                    await event.respond(f"🔍 Tracked keyword detected: '{keyword}'")
        
       
        for question, answer in faqs.items():
            if message_text and question in message_text:
                await event.respond(f"❓ FAQ Match:\n{answer}")
        
       
       
        
    except Exception as e:
        logger.error(f"Error processing message content: {e}")

async def summarize_target(event, target_type="chat"):
    try:
        chat_id = event.chat_id
        messages = []
        target_name = None
        entity = None

        if target_type in ["user", "group"]:
            args = event.message.text.split(maxsplit=1)
            if len(args) < 2:
                await event.respond(f"Usage: /summarize_{target_type} <{'username' if target_type == 'user' else 'group_name'}>")
                return

            target_name = args[1].strip()
            processing_msg = await event.respond(f"Finding {target_type} and generating summary, please wait...")

            try:
                entity = await client.get_entity(target_name)
                if not entity:
                    await processing_msg.edit(f"{target_type.capitalize()} '{target_name}' not found.")
                    return

                logger.info(f"Found {target_type} entity: {entity.id} ({type(entity).__name__})")

                if target_type == "user":
                    user_id = entity.id
                    messages = get_messages_by_sender(chat_id, user_id)
                    summary_title = f"Summary of messages from {target_name}"
                else:  # group
                    group_id = entity.id
                    messages = get_messages_from_db(group_id)
                    summary_title = f"Summary of messages in {target_name}"
            except Exception as e:
                logger.error(f"Error finding {target_type} '{target_name}': {e}")
                await processing_msg.edit(f"Error finding {target_type} '{target_name}'. Please check the username/group name and try again.")
                return
        else:  # chat
            processing_msg = await event.respond("Generating summary of this chat, please wait...")
            messages = get_messages_from_db(chat_id)
            summary_title = "Summary of recent messages"

        if not messages:
            await processing_msg.edit(f"No messages found for {target_type}.")
            return

        all_text = " ".join(messages)

        # Send to OpenServ for summarization
        openserv_response = send_to_openserv(all_text)
        summary = openserv_response.get("summary", "Could not generate summary.") if openserv_response else "Error fetching summary from OpenServ."

        summary_report = f"📊 {summary_title} 📊\n\n🔍 Summary:\n{summary}"

        await processing_msg.edit(summary_report)
        logger.info(f"Summary for {target_type} {target_name if target_name else chat_id} sent successfully")

    except Exception as e:
        logger.error(f"Error in summarize_target_openserv: {e}")
        await event.respond("❌ Error generating summary. Please try again.")

@client.on(events.NewMessage)
async def handle_new_message(event):
    try:
       
        if event.message.from_id and hasattr(event.message.from_id, 'bot') and event.message.from_id.bot:
            return
        
        chat_id = event.chat_id
        logger.info(f"New message in chat {chat_id}")
        
       
        if event.message.text and event.message.text.startswith('/'):
            logger.info(f"Command detected: {event.message.text}")
            return 
        
       
        chat_messages[chat_id].append(event.message)
        if len(chat_messages[chat_id]) > 1000:
            chat_messages[chat_id] = chat_messages[chat_id][-1000:]
        
       
        save_message_to_db(chat_id, event.message)
        
       
        await process_message_content(event)
        
    except Exception as e:
        logger.error(f"Error handling new message: {e}")

@client.on(events.NewMessage(pattern='/start'))
async def start_command(event):
    welcome_message = (
        "👋 Welcome to the Telegram AI Assistant!\n\n"
        "I can help you manage information overload in your chats. Here's what I can do:\n\n"
        "📝 /summarize - Get a summary of recent messages in this chat\n"
        "📝 /summarize_last <number> - Summarize the last N messages in chat\n"
        "👥 /summarize_group <group_name> - Summarize messages from a specific group\n"
         "🌐 /translate [target_lang] [text] - Translate text to target language\n"
        "🌐 /translate [source_lang] [target_lang] [text] - Translate from source to target language\n"
        "🌐 /languages - Show available language codes for translation\n"
        "🔍 /track <keyword> - Track important keywords\n"
        "❌ /untrack <keyword> - Stop tracking a keyword\n"
        "📊 /stats - Get chat statistics\n"
        "❓ /help - Show this help message\n\n"
        "Add me to a group chat to start managing information!"
    )
    await event.respond(welcome_message)

@client.on(events.NewMessage(pattern='/track'))
async def track_command(event):
    try:
       
        message_parts = event.message.text.split(maxsplit=1)
        if len(message_parts) < 2:
            await event.respond("Please specify a keyword to track. Usage: /track <keyword>")
            return
        
        keyword = message_parts[1].lower()
        chat_id = event.chat_id
        
       
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tracked_keywords (chat_id, keyword, added_by) VALUES (?, ?, ?)",
            (chat_id, keyword, event.sender_id)
        )
        conn.commit()
        conn.close()
        
       
        tracked_keywords[chat_id].add(keyword)
        
        await event.respond(f"✅ Now tracking keyword: '{keyword}'")
        
    except Exception as e:
        logger.error(f"Error handling track command: {e}")
        await event.respond("Sorry, there was an error adding the keyword. Please try again.")

@client.on(events.NewMessage(pattern='/untrack'))
async def untrack_command(event):
    try:
       
        message_parts = event.message.text.split(maxsplit=1)
        if len(message_parts) < 2:
            await event.respond("Please specify a keyword to untrack. Usage: /untrack <keyword>")
            return
        
        keyword = message_parts[1].lower()
        chat_id = event.chat_id
        
       
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM tracked_keywords WHERE chat_id = ? AND LOWER(keyword) = ?",
            (chat_id, keyword)
        )
        conn.commit()
        conn.close()
        
       
        if chat_id in tracked_keywords:
            tracked_keywords[chat_id].discard(keyword)
        
        await event.respond(f"❌ Stopped tracking keyword: '{keyword}'")
        
    except Exception as e:
        logger.error(f"Error handling untrack command: {e}")
        await event.respond("Sorry, there was an error removing the keyword. Please try again.")

@client.on(events.NewMessage(pattern='/stats'))
async def stats_command(event):
    try:
        chat_id = event.chat_id
        
       
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
       
        cursor.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
        message_count = cursor.fetchone()[0]
        
       
        cursor.execute("SELECT COUNT(*) FROM urls WHERE chat_id = ?", (chat_id,))
        url_count = cursor.fetchone()[0]
        
       
        cursor.execute("SELECT keyword FROM tracked_keywords WHERE chat_id = ?", (chat_id,))
        keywords = [row[0] for row in cursor.fetchall()]
        
       
        yesterday = datetime.now() - timedelta(days=1)
        cursor.execute("""
            SELECT COUNT(DISTINCT sender_id) 
            FROM messages 
            WHERE chat_id = ? AND timestamp > ?
        """, (chat_id, yesterday))
        active_users = cursor.fetchone()[0]
        
        conn.close()
        
       
        stats_message = (
            "📊 Chat Statistics 📊\n\n"
            f"📝 Total Messages: {message_count}\n"
            f"🔗 Shared URLs: {url_count}\n"
            f"👥 Active Users (24h): {active_users}\n"
            f"🔍 Tracked Keywords: {len(keywords)}\n"
        )
        
        if keywords:
            stats_message += f"\nTracked Keywords List:\n" + "\n".join(f"• {kw}" for kw in keywords)
        
        await event.respond(stats_message)
        
    except Exception as e:
        logger.error(f"Error handling stats command: {e}")
        await event.respond("Sorry, there was an error getting the statistics. Please try again.")
from googletrans import Translator
@client.on(events.NewMessage(pattern=r'/summarize_last(?:@\w+)?'))
async def summarize_last_command(event):
    try:
        chat_id = event.chat_id
        message_parts = event.message.text.split(maxsplit=1)
        
       
        if len(message_parts) < 2:
            await event.respond("Usage: /summarize_last <number_of_messages> (e.g., /summarize_last 50)")
            return
        
       
        try:
            n_messages = int(message_parts[1])
            if n_messages <= 0:
                await event.respond("Please provide a positive number of messages to summarize.")
                return
        except ValueError:
            await event.respond("Please provide a valid number of messages to summarize.")
            return
        processing_msg = await event.respond(f"Generating summary of the last {n_messages} messages, please wait...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, n_messages)
        )
        messages = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        if not messages:
            await processing_msg.edit("No messages found in this chat. Please ensure the bot has been here long enough to capture messages.")
            return
        all_text = " ".join(messages)
        summary = summarize_text(all_text)
        keywords = extract_keywords(all_text)
        keywords_str = ", ".join([f"{word} ({count})" for word, count in keywords])

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT url, timestamp 
            FROM urls 
            WHERE chat_id = ? 
            ORDER BY timestamp DESC LIMIT 5
        """, (chat_id,))
        urls_with_timestamps = cursor.fetchall()
        conn.close()
        
        urls_str = ""
        if urls_with_timestamps:
            urls_str = "🔗 Shared Links:\n"
            for url, timestamp in urls_with_timestamps:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    urls_str += f"• {url} (Posted: {formatted_time})\n"
                except:
                    urls_str += f"• {url}\n"
        else:
            urls_str = "🔗 Shared Links: None found"
        
        summary_report = (
            f"📊 Summary of Last {n_messages} Messages 📊\n\n"
            f"🔍 Key Points:\n{summary}\n\n"
            f"📈 Trending Keywords:\n{keywords_str}\n\n"
            f"{urls_str}"
        )
        
        await processing_msg.edit(summary_report)
        
    except Exception as e:
        logger.error(f"Error in summarize_last command: {e}")
        await event.respond(f"Sorry, there was an error generating the summary: {str(e)}")
@client.on(events.NewMessage(pattern=r'/translate(?:@\w+)?'))
async def translate_command(event):
    try:        
        message_parts = event.message.text.split(maxsplit=3)
        reply_to_msg = await event.get_reply_message()
        
        if reply_to_msg and reply_to_msg.text:
            if len(message_parts) == 1:
                source_lang = "auto"
                target_lang = "en"
                text = reply_to_msg.text
            elif len(message_parts) == 2:
                source_lang = "auto"
                target_lang = message_parts[1].lower()
                text = reply_to_msg.text
            elif len(message_parts) == 3:
                source_lang = message_parts[1].lower()
                target_lang = message_parts[2].lower()
                text = reply_to_msg.text
            else:
                await event.respond("Invalid format. Use: /translate [target_language] or /translate [source_language] [target_language] when replying to a message.")
                return
        else:
            if len(message_parts) < 3:
                await event.respond("Please use the format: /translate [target_language] [text] or reply to a message with /translate [target_language]")
                return
            elif len(message_parts) == 3:
                source_lang = "auto"
                target_lang = message_parts[1].lower()
                text = message_parts[2]
            elif len(message_parts) >= 4:
                source_lang = message_parts[1].lower()
                target_lang = message_parts[2].lower()
                text = message_parts[3]
        
        processing_msg = await event.respond("Translating, please wait...")
        try:
            translation = translator.translate(text, src=source_lang, dest=target_lang)
            
            response = (
                f"🌐 Translation\n\n"
                f"From: {translation.src} ({translation.src.upper()})\n"
                f"To: {translation.dest} ({translation.dest.upper()})\n\n"
                f"Original: {text[:50]}{'...' if len(text) > 50 else ''}\n\n"
                f"Translation: {translation.text}"
            )
            await processing_msg.edit(response)
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            await processing_msg.edit(f"Error during translation: {str(e)}\n\nMake sure the language codes are valid (e.g., 'en' for English, 'es' for Spanish, 'fr' for French, etc.)")
    
    except Exception as e:
        logger.error(f"Error in translate command: {e}")
        await event.respond("Sorry, there was an error processing your translation request.")

@client.on(events.NewMessage(pattern='/languages'))
async def languages_command(event):
    languages = {
        "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic", "hy": "Armenian",
        "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian", "bn": "Bengali", "bs": "Bosnian",
        "bg": "Bulgarian", "ca": "Catalan", "ceb": "Cebuano", "ny": "Chichewa", "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)", "co": "Corsican", "hr": "Croatian", "cs": "Czech", "da": "Danish",
        "nl": "Dutch", "en": "English", "eo": "Esperanto", "et": "Estonian", "tl": "Filipino", "fi": "Finnish",
        "fr": "French", "fy": "Frisian", "gl": "Galician", "ka": "Georgian", "de": "German", "el": "Greek",
        "gu": "Gujarati", "ht": "Haitian Creole", "ha": "Hausa", "haw": "Hawaiian", "iw": "Hebrew", "hi": "Hindi",
        "hmn": "Hmong", "hu": "Hungarian", "is": "Icelandic", "ig": "Igbo", "id": "Indonesian", "ga": "Irish",
        "it": "Italian", "ja": "Japanese", "jw": "Javanese", "kn": "Kannada", "kk": "Kazakh", "km": "Khmer",
        "ko": "Korean", "ku": "Kurdish (Kurmanji)", "ky": "Kyrgyz", "lo": "Lao", "la": "Latin", "lv": "Latvian",
        "lt": "Lithuanian", "lb": "Luxembourgish", "mk": "Macedonian", "mg": "Malagasy", "ms": "Malay",
        "ml": "Malayalam", "mt": "Maltese", "mi": "Maori", "mr": "Marathi", "mn": "Mongolian", "my": "Myanmar (Burmese)",
        "ne": "Nepali", "no": "Norwegian", "ps": "Pashto", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
        "pa": "Punjabi", "ro": "Romanian", "ru": "Russian", "sm": "Samoan", "gd": "Scots Gaelic", "sr": "Serbian",
        "st": "Sesotho", "sn": "Shona", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
        "so": "Somali", "es": "Spanish", "su": "Sundanese", "sw": "Swahili", "sv": "Swedish", "tg": "Tajik",
        "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
        "uz": "Uzbek", "vi": "Vietnamese", "cy": "Welsh", "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba",
        "zu": "Zulu"
    }
    
    lang_message = "🌐 Available Language Codes for Translation 🌐\n\n"
    sorted_langs = sorted(languages.items(), key=lambda x: x[1])
    for i, (code, name) in enumerate(sorted_langs):
        lang_message += f"{code}: {name}"
        if i % 2 == 0:
            lang_message += " | "
        else:
            lang_message += "\n"
    await event.respond(lang_message)
@client.on(events.NewMessage(pattern='/help'))
async def help_command(event):
    try:
        help_message = (
            "🤖 Telegram AI Assistant Help 🤖\n\n"
            "Available Commands:\n\n"
            "📝 /summarize - Get a summary of current chat\n"
            "📊 /summarize_chat <chat_name> [days] - Get detailed summary of specific chat\n"
            "🔍 /track <keyword> - Track important keywords\n"
            "❌ /untrack <keyword> - Stop tracking a keyword\n"
            "📊 /stats - Get chat statistics\n"
            "🌐 /translate [target_lang] [text] - Translate text to target language\n"
            "🌐 /translate [source_lang] [target_lang] [text] - Translate from source to target language\n"
            "🌐 /languages - Show available language codes for translation\n"
            "❓ /help - Show this help message\n\n"
            "Translation Tips:\n"
            "• Reply to a message with /translate [target_lang] to translate it\n"
            "• Use 'auto' as source language for automatic detection\n"
            "• Example: /translate es Hello world (translates to Spanish)\n"
            "• Example: /translate auto fr Hola mundo (detects Spanish and translates to French)\n\n"
            "Usage Tips:\n"
            "• Add me to a group chat to start collecting messages\n"
            "• Use /track to get notified about important topics\n"
            "• Use /summarize periodically to catch up on discussions\n"
            "• Use /summarize_chat @groupname 7 to get a 7-day summary\n"
            "• Check /stats to see chat activity\n\n"
            "For more help, contact: @lalithkumar11 @"
        )
        await event.respond(help_message)

    except Exception as e:
        logger.error(f"Error handling help command: {e}")
        await event.respond("Sorry, there was an error displaying the help message. Please try again later.")

async def get_chat_by_name(chat_name):
    try:
       
        chat = await client.get_entity(chat_name)
        return chat
    except Exception as e:
        logger.error(f"Error finding chat {chat_name}: {e}")
        return None

async def extract_media_info(message):
    media_info = []
    try:
        if message.media:
            if hasattr(message.media, 'photo'):
                media_info.append("📷 Photo")
            elif hasattr(message.media, 'document'):
                media_info.append(f"📎 Document: {message.media.document.mime_type}")
            elif hasattr(message.media, 'video'):
                media_info.append("🎥 Video")
    except Exception as e:
        logger.error(f"Error extracting media info: {e}")
    return media_info

@client.on(events.NewMessage(pattern=r'/summarize(?:@\w+)?$'))
async def summarize_command(event):
    await summarize_target(event, "chat")

@client.on(events.NewMessage(pattern=r'/summarize_user(?:@\w+)?'))
async def summarize_user_command(event):
    await summarize_target(event, "user")

@client.on(events.NewMessage(pattern=r'/summarize_group(?:@\w+)?'))
async def summarize_group_command(event):
    await summarize_target(event, "group")
@client.on(events.NewMessage(pattern='/force_update'))
async def force_update_command(event):
    try:
        chat_id = event.chat_id
        processing_msg = await event.respond("Force updating database with recent messages, please wait...")
        messages = await client.get_messages(chat_id, limit=100)
        count = 0
        for message in messages:
            save_message_to_db(chat_id, message)
            count += 1
        await processing_msg.edit(f"✅ Database updated with {count} recent messages.")
    except Exception as e:
        logger.error(f"Error in force update command: {e}")
        await event.respond(f"Error updating database: {str(e)}")
async def check_bot_permissions(chat_id):
    try:
        chat = await client.get_entity(chat_id)
        if isinstance(chat, (types.Chat, types.Channel)):
            permissions = await client.get_permissions(chat_id, 'me')
            required_permissions = {
                'send_messages': True,
                'read_messages': True,
            }
            missing_permissions = []
            for perm, required in required_permissions.items():
                if not getattr(permissions, perm, False):
                    missing_permissions.append(perm)
            return not missing_permissions, missing_permissions
        else:
            logger.error(f"Chat {chat_id} is not a group or channel.")
            return False, ["not_a_group_or_channel"]
    except Exception as e:
        logger.error(f"Error checking permissions: {e}")
        return False, ["unknown"]
async def check_bot_permissions(chat_id):
    try:
        chat = await client.get_entity(chat_id)
        if isinstance(chat, (types.Chat, types.Channel)):
            permissions = await client.get_permissions(chat_id, 'me') 
            required_permissions = {
                'send_messages': True,
                'read_messages': True,
            }
            missing_permissions = []
            for perm, required in required_permissions.items():
                if not getattr(permissions, perm, False):
                    missing_permissions.append(perm)
            return not missing_permissions, missing_permissions
        else:
            logger.error(f"Chat {chat_id} is not a group or channel.")
            return False, ["not_a_group_or_channel"]
    except Exception as e:
        logger.error(f"Error checking permissions: {e}")
        return False, ["unknown"]
async def main():
    try:
        init_db()
        logger.info("Database initialized successfully")
        load_tracked_keywords()
        load_faqs_from_db()
        logger.info("Data loaded successfully")
        logger.info("Starting bot with credentials:")
        logger.info(f"API_ID: {'Set' if API_ID else 'Not Set'}")
        logger.info(f"API_HASH: {'Set' if API_HASH else 'Not Set'}")
        logger.info(f"BOT_TOKEN: {'Set' if BOT_TOKEN else 'Not Set'}")
        await client.start(bot_token=BOT_TOKEN)
        me = await client.get_me()
        logger.info(f"Bot started successfully as @{me.username} (ID: {me.id})")
        await client.run_until_disconnected()
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        logger.error("Please check your .env file and make sure all credentials are set correctly")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        await client.disconnect()
if __name__ == "__main__":
    asyncio.run(main())