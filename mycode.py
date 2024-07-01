import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
# Download NLTK resources
nltk.download('vader_lexicon')

# Function to process WhatsApp chat file
def process_whatsapp_chat(uploaded_file):
    # Read the WhatsApp chat file line by line
    conversation = uploaded_file.readlines()

    # Define functions to process each line of the chat
    def date_time(s):
        pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)(?:\s?(am|pm|AM|PM))? -'
        result = re.match(pattern, s)
        if result:
            return True
        return False

    def find_author(s):
        s = s.split(":")
        if len(s) == 2:
            return True
        else:
            return False

    def messages(line):
        splitline = line.split(' - ')
        dateTime = splitline[0]
        date, time = dateTime.split(",")
        message = " ".join(splitline[1:])

        if find_author(message):
            splitmessage = message.split(": ")
            author = splitmessage[0]
            message = " ".join(splitmessage[1:])
        else:
            author = None

        return date, time, author, message

    # Process each line of the conversation
    data = []
    messageBuffer = []
    for line in conversation:
        line = line.decode("utf-8").strip()  # Decode bytes to string and remove leading/trailing whitespace
        if date_time(line):
            if len(messageBuffer) > 0:
                data.append([date, time, author, ' '.join(messageBuffer)])
            messageBuffer.clear()
            date, time, author, message = messages(line)
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)

    # Create a DataFrame from the processed data
    df = pd.DataFrame(data, columns=["Date", 'Time', 'Author', 'Message'])
    df['Date'] = pd.to_datetime(df['Date'])

    # Add day of the week column
    df['Day_of_week'] = df['Date'].dt.day_name()

    # Add month column
    df['Month'] = df['Date'].dt.month_name()

    # Remove rows with missing values
    data = df.dropna()

    # Perform sentiment analysis
    sentiments = SentimentIntensityAnalyzer()
    data["Positive"] = data["Message"].apply(lambda msg: sentiments.polarity_scores(msg)["pos"])
    data["Negative"] = data["Message"].apply(lambda msg: sentiments.polarity_scores(msg)["neg"])
    data["Neutral"] = data["Message"].apply(lambda msg: sentiments.polarity_scores(msg)["neu"])

    return data

# Function to clean text
def clean_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to extract emojis from messages
def extract_emojis(text):
    emoji_list = []
    emoji_regex = r'[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001F004\U0001F0CF\U0001F170-\U0001F251\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+'
    emojis = re.findall(emoji_regex, text)
    for emoji in emojis:
        emoji_list.append(emoji)
    return emoji_list


# Streamlit app
def main():
    st.title('WhatsApp Chat Analysis')

    # File upload
    st.sidebar.header('Upload WhatsApp Chat Export File')
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Process the WhatsApp chat file
        data = process_whatsapp_chat(uploaded_file)

        # Total number of messages
        total_messages = data.shape[0]

        # Total number of users
        total_users = data['Author'].nunique()

        # Total number of words
        total_words = data['Message'].apply(lambda x: len(x.split())).sum()

        # Total number of media shared
        total_media = data['Message'].apply(lambda x: "<Media omitted>" in x).sum()

        st.subheader('Chat Summary')
        st.write(f"Total number of messages: {total_messages}")
        st.write(f"Total number of users: {total_users}")
        st.write(f"Total number of words: {total_words}")
        st.write(f"Total number of media shared: {total_media}")

        # Chat Analysis
        st.subheader('Messages per Author')
        st.write(data.groupby("Author")["Message"].count())

        st.subheader('Average Message Length per Author')
        st.write(data.groupby("Author")["Message"].apply(lambda x: x.str.len().mean()))

        st.subheader('Total Messages per Date')
        messages_per_date = data.groupby(data["Date"].dt.date)["Message"].count()
        
        st.line_chart(messages_per_date)
        st.subheader('Total Messages per Date')
        st.write(messages_per_date)
        
        # Clean the text
        data['Clean_Message'] = data['Message'].apply(clean_text)

        # Split each message into words
        words = data['Clean_Message'].str.split(expand=True).unstack().value_counts()

        # Display the most common words in the chat
        st.subheader('Most Common Words in the Chat:')
        st.write(words.head(10))


        # Word Frequency Analysis
        st.subheader('Word Frequency Analysis')
        words = data['Message'].str.lower().str.split().explode().value_counts()
        st.bar_chart(words.head(20))

        # Word Cloud Visualization
        st.subheader('Word Cloud Visualization')
        text = " ".join(message for message in data['Message'].str.lower())
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(text)
        st.image(wordcloud.to_array(), caption='Word Cloud')

        # Extracting emojis from messages
        if 'Message' in data.columns:
            data['Emojis'] = data['Message'].apply(extract_emojis)

            # Flattening the list of emojis
            emojis_list = [emoji for sublist in data['Emojis'] for emoji in sublist]

            # Counting the occurrence of each emoji
            emoji_counts = Counter(emojis_list)

            # Getting the most common emojis
            most_common_emojis = emoji_counts.most_common(10)

            # Plotting the most used emojis
            if most_common_emojis:
                emojis, frequencies = zip(*most_common_emojis)
            else:
                emojis, frequencies = [], []

            plt.figure(figsize=(10, 6))
            plt.bar(emojis, frequencies)
            plt.title('Top 10 Emojis Used')
            plt.xlabel('Emoji')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            fig, ax = plt.gcf(), plt.gca()
            st.pyplot(fig)

        # Sentiment Analysis
        st.subheader('Sentiment Analysis Results')
        st.write(data.head(20))

        # Plot sentiment analysis
        plot_sentiment_analysis(data)

        # Top 10 users with the most number of messages
        st.subheader('Top 10 Users with Maximum Number of Messages')
        top_10_users = data['Author'].value_counts().head(10)
        st.bar_chart(top_10_users)

        # Grouping data by date
        st.subheader('Number of Messages by Date')
        messages_by_date = data.groupby(data['Date'].dt.date).size()
        st.line_chart(messages_by_date)

        # Finding the days with the most number of messages
        st.write("\n")
        st.write(f"The day with the maximum number of messages was {messages_by_date.idxmax()} with {messages_by_date.max()} messages.")

        # Plotting number of messages by date
        plt.figure(figsize=(14, 6))
        messages_by_date.plot(kind='line', color='skyblue')
        plt.title('Number of Messages by Date')
        plt.xlabel('Date')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45)
        plt.grid(True)
        fig, ax = plt.gcf(), plt.gca()
        st.pyplot(fig)

        # Plotting the top 10 users
        st.subheader('Top 10 Users with Maximum Number of Messages')
        plt.figure(figsize=(10, 6))
        top_10_users.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Users with Maximum Number of Messages')
        plt.xlabel('Users')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        fig, ax = plt.gcf(), plt.gca()
        st.pyplot(fig)

        # Plotting most busy day in a week
        st.subheader('Most Busy Day in a Week')
        plt.figure(figsize=(10, 6))
        data['Day_of_week'].value_counts().plot(kind='bar', color='skyblue')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45)
        fig, ax = plt.gcf(), plt.gca()
        st.pyplot(fig)

        # Plotting most busy month
        st.subheader('Most Busy Month')
        fig, ax = plt.subplots(figsize=(10, 6))
        data['Month'].value_counts().plot(kind='bar', color='salmon', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Messages')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig, ax = plt.gcf(), plt.gca()
        st.pyplot(fig)

# Plot sentiment analysis
def plot_sentiment_analysis(data):
    x = sum(data["Positive"])
    y = sum(data["Negative"])
    z = sum(data["Neutral"])

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [x, y, z]
    colors = ['gold', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')
    ax.set_title('Sentiment Analysis')
    st.pyplot(fig)

# Run the app
if __name__ == '__main__':
    main()
