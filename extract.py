from datetime import datetime, timezone
from apify_client import ApifyClient
from apify_client._errors import ApifyApiError
import os
import json

from dotenv import load_dotenv

load_dotenv()

client = ApifyClient(os.environ.get("APIFY_API_KEY"))

def extract_fields_twitter(data, num_search, follow_count=0):
    """Extracts tweet details including ID, URL, caption, media, timestamp, and username."""
    extracted = []

    if num_search is None:
        num_search = 10
    elif isinstance(num_search, str):
        try:
            num_search = int(num_search)
        except ValueError:
            raise ValueError(f"Invalid num_search value: {num_search}. Must be an integer.")
    if follow_count is None:
        follow_count = 200
    elif isinstance(follow_count, str):
        try:
            follow_count = int(follow_count)
        except ValueError:
            raise ValueError(f"Invalid follow_count value: {follow_count}. Must be an integer.")    

    if isinstance(data, dict):
        tweet_id = data.get("id", "")
        text = data.get("text", "")
        timestamp = data.get("createdAt", "")
        username = data.get("author", {}).get("userName", "")
        id_username = data.get("author", {}).get("id", "")
        url = data.get("url", "")
        media_data = data.get("extendedEntities", {}).get("media", [])
        media_urls = [media.get("expanded_url", "") for media in media_data if "expanded_url" in media]
        
        comments = []
        retweets = []

        if tweet_id:
            input_query = {
                "conversation_ids": [tweet_id],
                # "max_items_per_conversation": num_search
            }
            print(f"🔄 Scraping replies for tweet ID: {tweet_id}...")

            try:
                run = client.actor("kaitoeasyapi/twitter-reply").call(run_input=input_query)
                replies = list(client.dataset(run["defaultDatasetId"]).iterate_items(limit=10))
            except ApifyApiError as e:
                if "Monthly usage hard limit exceeded" in str(e):
                    print("🚨 Apify API limit reached! Skipping this request.")
                    replies = []
                else:
                    raise e  

            for reply in replies:
                reply_tweet_id = reply.get("id", "")
                reply_text = reply.get("text", "")
                reply_timestamp = reply.get("createdAt", "")
                reply_username = reply.get("author", {}).get("userName", "")
                reply_id_username = reply.get("author", {}).get("id", "")

                reply_media_urls = []
                if "extendedEntities" in reply and "media" in reply["extendedEntities"]:
                    reply_media_urls = [media.get("media_url_https", "") for media in reply["extendedEntities"]["media"]]

                comments.append({
                    "id_post": reply_tweet_id,
                    "caption": reply_text,
                    "media": reply_media_urls,
                    "timestamp": reply_timestamp,
                    "username": reply_username,
                    "id_username": reply_id_username
                })
            print(f"✅ Finished scraping replies for tweet ID: {tweet_id}") 
            print(f"🔄 Scraping retweets for tweet ID: {tweet_id}...")
            input_query_retweet = {
                "tweet_ids": [
                    tweet_id
                ],
            }
            try:
                run = client.actor("kaitoeasyapi/tweet-reweet-userlist").call(run_input=input_query_retweet)
                retweet = list(client.dataset(run["defaultDatasetId"]).iterate_items(limit=10))
            except ApifyApiError as e:
                if "Monthly usage hard limit exceeded" in str(e):
                    print("🚨 Apify API limit reached! Skipping this request.")
                    retweet = []
                else:
                    raise e    
            for retweets_user in retweet:
                retweet_type = retweets_user.get("type", "")
                retweet_id_username = retweets_user.get("id", "")
                retweet_url = retweets_user.get("url", "")
                retweet_twitterUrl = retweets_user.get("twitterUrl", "")
                retweet_username = retweets_user.get("userName", "")
                retweet_isverified = retweets_user.get("isBlueVerified","")
                retweet_name = retweets_user.get("name","")
                retweet_description = retweets_user.get("description","")
                retweet_timestamp = retweets_user.get("createdAt","")   

                retweets.append({
                    "type": retweet_type,
                    "id_username": retweet_id_username,
                    "url": retweet_url,
                    "twitterUrl": retweet_twitterUrl,
                    "username": retweet_username,
                    "isverified": retweet_isverified,
                    "name": retweet_name,
                    "description": retweet_description,
                    "timestamp": retweet_timestamp
                })   
            print(f"✅ Finished scraping retweets for tweet ID: {tweet_id}")

        extracted.append({
            "type": "tweet",
            "id_post": tweet_id,
            "url": url,
            "username": username,
            "id_username": id_username,
            "caption": text,
            "media": media_urls,
            # "relations": relations,
            "retweets": retweets,
            "comments": comments,
            "timestamp": timestamp
        })

    elif isinstance(data, list):
        for item in data:
            extracted.extend(extract_fields_twitter(item, num_search, follow_count))

    return extracted