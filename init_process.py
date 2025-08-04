import pandas as pd
from scrp import twitterScrapping
from extract import extract_fields_twitter
import json
import torch

# from env import engine
# from sqlalchemy.types import ARRAY, Text

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import timezone, datetime
import uuid
import argparse
import ast

from huggingface_hub import login

def load_sentiment_model():
    print("Loading sentiment model...")

    sentiment_model_path = "mdhugol/indonesia-bert-sentiment-classification"
    model_sentiment = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
    tokenizer_sentiment = AutoTokenizer.from_pretrained(sentiment_model_path)
    print("Sentiment model loaded!")
    
    return model_sentiment, tokenizer_sentiment

def load_emotion_model():

    print("Loading emotion model...")

    login(token="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")
    
    emotion_pipe = pipeline(
            "text-classification",
            model="Rqwannn/indoBERT_emotion_model",
            tokenizer="Rqwannn/indoBERT_emotion_model",
        )
    
    print("Emotion model loaded!")

    return emotion_pipe

def ensure_list_of_strings(value):
    """Ensures the value is a list of strings, compatible with PostgreSQL TEXT[]"""
    if isinstance(value, str):
        try:
            value = json.loads(value)  # Convert from JSON string
        except json.JSONDecodeError:
            return [value]  # If it's a plain string, wrap in a list
    if isinstance(value, dict):
        return [json.dumps(value)]  # Convert dict to string inside a list
    if isinstance(value, list):
        return [str(v) for v in value]  # Ensure all elements are strings
    return [str(value)]  # Default case: Convert to list with a single string item

def ensure_list_of_dicts(value):
    if isinstance(value, str):
        import ast
        value = ast.literal_eval(value)
    if not isinstance(value, list):
        value = [value]
    return [v for v in value if isinstance(v, dict)]


def startProcess(user_uuid, search_query, query_type, num_search, source, follow_count, start_date, until_date):
    """Runs the entire data extraction, sentiment analysis, and storage process."""
    
    model_sentiment, tokenizer_sentiment = load_sentiment_model()
    emotion_pipe = load_emotion_model()

    startDate = datetime.strptime(start_date, "%Y-%m-%d_%H:%M:%S")
    endDate = datetime.strptime(until_date, "%Y-%m-%d_%H:%M:%S")

    if source == 'x':
        start_date_x = datetime.strftime(startDate, "%Y-%m-%d_%H:%M:%S_UTC")
        end_date_x = datetime.strftime(endDate, "%Y-%m-%d_%H:%M:%S_UTC")
        raw_data = twitterScrapping(search_query, num_search, start_date_x, end_date_x, query_type)
    else:
        print('🚨 Source not found')
        return pd.DataFrame()

    if not raw_data:
        print("❌ No data found.")
        return pd.DataFrame()

    cleaned_data = extract_fields_twitter(raw_data, num_search, follow_count)

    if not cleaned_data:
        print("❌ No relevant fields found in the data.")
        return pd.DataFrame()

    df = pd.DataFrame(cleaned_data)

    df["media"] = df["media"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    df["comments"] = df["comments"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    df['retweets'] = df['retweets'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    user_uuid = user_uuid
    search_query = search_query
    id_post = df.get("id_post", []).tolist()
    captions = df.get("caption", []).tolist()
    urls = df.get("url", []).tolist()
    timestamps = df.get("timestamp", []).tolist()
    usernames = df.get("username", []).tolist()
    id_username = df.get("id_username", []).tolist()
    types = df.get("type", []).tolist()
    media_list = df.get("media", []).tolist()
    comments_list = df.get("comments", []).tolist()
    retweets_list = df.get("retweets", []).tolist()

    results = []

    def iso_timestamp():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    current_time = iso_timestamp()

    comment_records = []
    retweet_records = []

    for id_p, caption, url, timestamp, username, media, id_u, type_, comments, retweets in zip(
            id_post, captions, urls, timestamps, usernames, media_list, id_username, types, comments_list, retweets_list):
        try:
            caption = caption or ""  
            username = username or ""
            timestamp = timestamp or ""
            url = url or ""
            id_u = id_u or ""

            inputs_sentiment = tokenizer_sentiment(
                caption, return_tensors="pt", truncation=True, padding=True, max_length=505
            )

            with torch.no_grad():
                outputs_sentiment = model_sentiment(**inputs_sentiment)

            logits_sentiment = outputs_sentiment.logits
            probabilities = torch.nn.functional.softmax(logits_sentiment, dim=-1)
            label_idx = torch.argmax(probabilities, dim=-1).item()

            label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment_label = label_mapping.get(label_idx, "unknown")

            emotion_result = emotion_pipe(caption)[0]
            emotion_label = emotion_result['label']  # Contoh: "MARAH"
        
            # Store results with additional fields
            sentiment_result = {
                "user_uuid": user_uuid, 
                "uuid": str(uuid.uuid4()),
                "source": source,
                "search_query": search_query,
                "type": type_,
                "id_post": id_p,
                "sentiment": sentiment_label,
                "emotion": emotion_label,
                "url": url,
                "id_username": id_u,
                "username": username,
                "caption": caption,
                "timestamp": timestamp,
                "media": ensure_list_of_strings(media),
                "status_attack": False,
                "createdAt": current_time,
                "updatedAt": current_time
            }

            results.append(sentiment_result)

            """
                Table: comments

                Kolom:
                - id_post: Foreign Key ke sentiment_results.id_post
                - caption: Text
                - media: ARRAY(String) atau JSONB
                - timestamp: DateTime
                - username: String
                - id_username: String
            """

            for com in ensure_list_of_strings(comments):
                com = ast.literal_eval(com)
                com_result = {
                    "caption": com["caption"],
                    "username": com["username"],
                    "timestamp": current_time,
                    "id_post": id_p
                }

                if com["caption"] is not None:
                    emotion_result = emotion_pipe(com["caption"])[0]
                    emotion_label = emotion_result['label'] 

                    # Perform sentiment analysis
                    inputs_sentiment = tokenizer_sentiment(com["caption"], return_tensors="pt", truncation=True, padding=True, max_length=505)

                    with torch.no_grad():
                        outputs_sentiment = model_sentiment(**inputs_sentiment)

                    logits_sentiment = outputs_sentiment.logits
                    probabilities = torch.nn.functional.softmax(logits_sentiment, dim=-1)
                    label_idx = torch.argmax(probabilities, dim=-1).item()

                    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
                    sentiment_label = label_mapping.get(label_idx, "unknown")

                else:
                    emotion_label = "-"
                    sentiment_label = "-"

                com_result["id_username"] = com.get("id_username", "-")
                com_result["media"] = com.get("media", "-")
                com_result["id_comments"] = str(uuid.uuid4())

                com_result["createdAt"] = current_time
                com_result["updatedAt"] = current_time

                com_result["emotion"] = emotion_label
                com_result["sentiment"] = sentiment_label
                comment_records.append(com_result)

            """
                Table: retweets

                Kolom:
                - type: String
                - id_username: BigInteger atau String
                - url: Text
                - twitterUrl: Text
                - username: String
                - isverified: String atau Boolean
                - name: String
                - description: Text
                - timestamp: DateTime atau String
                - id_post: Foreign Key ke sentiment_results.id_post
            """

            for rt in ensure_list_of_strings(retweets):
                rt = ast.literal_eval(rt)
                rt["id_post"] = id_p
                rt["is_verified"] = rt["isverified"]
                rt.pop("isverified")

                if rt["description"] is not None:
                    emotion_result = emotion_pipe(rt["description"])[0]
                    emotion_label = emotion_result['label'] 

                    # Perform sentiment analysis
                    inputs_sentiment = tokenizer_sentiment(rt["description"], return_tensors="pt", truncation=True, padding=True, max_length=505)

                    with torch.no_grad():
                        outputs_sentiment = model_sentiment(**inputs_sentiment)

                    logits_sentiment = outputs_sentiment.logits
                    probabilities = torch.nn.functional.softmax(logits_sentiment, dim=-1)
                    label_idx = torch.argmax(probabilities, dim=-1).item()

                    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
                    sentiment_label = label_mapping.get(label_idx, "unknown")

                else:
                    emotion_label = "-"
                    sentiment_label = "-"
                
                rt["createdAt"] = current_time
                rt["updatedAt"] = current_time
                rt["id_retweets"] = str(uuid.uuid4())
                rt["emotion"] = emotion_label
                rt["sentiment"] = sentiment_label
                retweet_records.append(rt)
        except Exception as e:
            continue
    
        df_results = pd.DataFrame(results)
        df_comments = pd.DataFrame(comment_records)
        df_retweets = pd.DataFrame(retweet_records)

        try:
            df_results.to_json("sentiment_results.json", orient="records", force_ascii=False, indent=2)
            df_comments.to_json("comments.json", orient="records", force_ascii=False, indent=2)
            df_retweets.to_json("retweets.json", orient="records", force_ascii=False, indent=2)

            print("✅ Semua data berhasil disimpan ke file JSON.")
        except Exception as e:
            print("❌ JSON save error:", e)

        all_data = {
            "sentiment_results": results,
            "comments": comment_records,
            "retweets": retweet_records
        }

        with open("all_data.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Start the scraping process")
    parser.add_argument("--user_uuid", required=True, help="User UUID")
    parser.add_argument("--search_query", required=True, help="Search query for scraping example 'username' or 'hashtag'")
    parser.add_argument("--query_type" , required=True, choices=["Top", "Latest"], help="Type of search query")
    parser.add_argument("--num_search", type=int, required=True, help="Number of results to scrape example usage 20")
    parser.add_argument("--source", required=True, choices=["instagram", "x", "facebook", "tiktok"], help="Data source")
    parser.add_argument("--follow_count", required=True, type=int, default=200, help="Number of followers to scrap")
    parser.add_argument("--start_date", required=True, help="Start date for scraping 'example usage 2001-04-20_00:00:00'")
    parser.add_argument("--until_date", default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), help="End date for scraping 'example usage 2005-09-11_00:00:00'")

    args = parser.parse_args()

    startProcess(args.user_uuid, args.search_query, args.query_type, args.num_search, args.source, args.follow_count, args.start_date, args.until_date, )


if __name__ == "__main__":
    main()