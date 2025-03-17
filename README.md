# Telegram AI Assistant Bot

A powerful AI-driven Telegram bot that can summarize chat messages, track keywords, extract important links, and provide translations. The bot integrates with OpenServ for AI-based text summarization and uses an SQLite database to store and manage messages.

## Problem Statement 📌
Telegram is a widely used messaging platform with millions of active users. However, in large group chats, trading communities, work discussions, and educational forums, managing and retrieving important information becomes a challenge.

Users often struggle with:
- Overwhelming message volumes, making it difficult to stay updated.
- Important messages getting buried under long discussions.
- Repetitive questions consuming time and effort.
- Lack of efficient summarization and data extraction tools.
- Difficulty in tracking engagement and trending topics in chats.

Due to these challenges, users spend significant time scrolling through messages, often missing crucial information. There is a clear need for an intelligent solution that filters relevant messages, summarizes discussions, and enhances the overall efficiency of Telegram interactions.

## Objective 🎯
The objective of this project is to develop an AI-powered Telegram bot that enhances user experience by:
✅ Summarizing long discussions for quick insights.
✅ Extracting key information such as links, important messages, and numbers.
✅ Providing real-time text translation across different languages.
✅ Tracking keyword mentions and notifying users about relevant discussions.
✅ Providing group analytics to monitor engagement and trending topics.

## Our Solution 💡
To address these challenges, we have developed an AI-powered Telegram assistant that acts as a smart filter for conversations. Our solution leverages Natural Language Processing (NLP) and AI-driven automation to extract, summarize, and organize important messages efficiently.

With our bot, users no longer need to scroll through hundreds of messages to find key updates. The bot provides real-time conversation summaries, highlights important messages, links, and numbers, and even automates responses to frequently asked questions. Additionally, it includes language translation capabilities, ensuring seamless communication across diverse communities.

For group admins and businesses, our bot tracks engagement, monitors trending topics, and provides actionable insights into group activity, helping to streamline discussions and improve collaboration.

By eliminating clutter, reducing redundancy, and enhancing accessibility, our solution transforms the way Telegram users interact, making conversations more structured, efficient, and user-friendly.

## Expected Outcome 🎯
By the end of development, this bot will:
✅ Provide instant summaries of long discussions.
✅ Extract and organize important messages and links.
✅ Automate responses to frequently asked questions.
✅ Improve group management through analytics and keyword tracking.
✅ Reduce chat clutter and enhance productivity for Telegram users.

## Features 🚀
- **Summarize Messages**: Generate summaries for users, groups, or entire chats.
- **Keyword Tracking**: Monitor specific words and get alerts when mentioned.
- **AI-Powered Summarization**: Uses OpenServ API for advanced text summarization.
- **Message Database**: Stores messages and extracts useful insights.
- **URL Extraction**: Identifies and tracks shared links.
- **Translation Support**: Supports multilingual translations with Google Translate.
- **Chat Statistics**: Provides chat activity insights.

## Prerequisites 🛠
- Python 3.8+
- Telegram API credentials
- OpenServ API key (for AI summarization)
- Required Python libraries

## Setup Instructions 🔧

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/telegram-ai-assistant.git
cd telegram-ai-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root and add your Telegram API credentials:
```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
BOT_TOKEN=your_bot_token
OPEN_SERV_API_KEY=your_openserv_api_key
OPEN_SERV_API_URL=https://api.openserv.com/summarize
```

### 4. Initialize the Database
```bash
python bot.py --init-db
```

### 5. Run the Bot
```bash
python bot.py
```

## Usage 📝

### Commands
| Command                  | Description |
|--------------------------|-------------|
| `/summarize`             | Summarize the last 50 messages in the chat. |
| `/summarize_user <user>` | Summarize messages from a specific user. |
| `/summarize_group <group>` | Summarize messages from a group. |
| `/track <keyword>`       | Track a specific keyword. |
| `/untrack <keyword>`     | Stop tracking a keyword. |
| `/stats`                 | View chat statistics. |
| `/translate <lang> <text>` | Translate text to a target language. |
| `/languages`             | Show supported translation languages. |
| `/help`                  | Display the help menu. |

## Troubleshooting ❓
### OpenServ API Not Responding
- Ensure the `OPEN_SERV_API_KEY` and `OPEN_SERV_API_URL` are correctly set.
- Check the OpenServ API status.

### Database Issues
- Run `python bot.py --init-db` to initialize the database.

### Bot Not Responding
- Ensure your bot token is correct.
- Check that the bot is added to the Telegram group.

## License 📜
This project is licensed under the MIT License.

## Contributors 💡
- Your Name (@your-telegram-handle)

## Contact 📬
For support, reach out to [your_email@example.com] or open an issue on GitHub.

