import os
import json
from apify_client import ApifyClient
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

client = ApifyClient(os.environ.get("APIFY_API_KEY"))
    
def twitterScrapping(search_query, num_search, init_date, end_date, query_type):
    try:
        # Prepare the Actor input
        input_query = {
            "filter:blue_verified": False,
            "filter:consumer_video": False,
            "filter:has_engagement": False,
            "filter:hashtags": False,
            "filter:images": False,
            "filter:links": False,
            "filter:media": False,
            "filter:mentions": False,
            "filter:native_video": False,
            "filter:nativeretweets": False,
            "filter:news": False,
            "filter:pro_video": False,
            "filter:quote": False,
            "filter:replies": False,
            "filter:safe": False,
            "filter:spaces": False,
            "filter:twimg": False,
            "filter:verified": False,
            "filter:videos": False,
            "filter:vine": False,
            "include:nativeretweets": False,
            "lang": "in",
            "maxItems": num_search,
            "queryType": query_type,
            "min_faves": 100,
            "min_replies": 100,
            "min_retweets": 100,
            "since": init_date,
            "twitterContent": search_query,
            "until": end_date
        }

        # Run the Actor and wait for it to finish
        print(f"Starting scrape for: {search_query} with limit {num_search}")
        run = client.actor("kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=input_query)
        print("Scraper finished running!")

        dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

        if not dataset_items:
            print("No data found!")
            return []

        json_filename = f"data/twitter_raw_{search_query}.json"
        os.makedirs("data", exist_ok=True)
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(dataset_items, f, indent=4, ensure_ascii=False)

        return dataset_items
    except Exception as e:
        print(f"Scraping error: {e}")
        return [] 