import os
import time
import pandas as pd
import logging
from dotenv import load_dotenv
from googleapiclient.discovery import build

# ------------------------------
# ✅ Logging Setup
# ------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/fetch_bulk.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------------
# ✅ Load API Key
# ------------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# ------------------------------
# 🔹 Step 1: Get video IDs by category
# ------------------------------
def get_video_ids(query, max_results=3):
    try:
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        )
        response = request.execute()
        return [item["id"]["videoId"] for item in response["items"]]
    except Exception as e:
        logging.error(f"❌ Error fetching video IDs for {query}: {e}")
        return []

# ------------------------------
# 🔹 Step 2: Fetch comments (with error handling)
# ------------------------------
def fetch_comments(video_id, category):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
    except Exception as e:
        logging.error(f"❌ Cannot fetch comments for {video_id} ({category}): {e}")
        return comments

    while request:
        try:
            response = request.execute()
        except Exception as e:
            if "commentsDisabled" in str(e):
                logging.warning(f"🚫 Comments disabled for video {video_id} ({category})")
                print(f"🚫 Comments disabled for video {video_id} ({category})")
                return comments
            else:
                logging.error(f"⚠️ API error for {video_id}: {e}, retrying in 5s...")
                print(f"⚠️ API error for {video_id}, retrying...")
                time.sleep(5)
                continue

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "category": category,
                "author": snippet.get("authorDisplayName"),
                "text": snippet.get("textDisplay"),
                "likes": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt")
            })

        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"],
                textFormat="plainText"
            )
        else:
            break

    return comments

# ------------------------------
# 🔹 Step 3: Main bulk fetch pipeline
# ------------------------------
if __name__ == "__main__":
    CATEGORIES = ["technology", "gaming", "movies", "music", "education"]
    all_comments = []

    for category in CATEGORIES:
        print(f"\n🔎 Fetching videos for {category}")
        logging.info(f"Fetching videos for category: {category}")

        video_ids = get_video_ids(category, max_results=3)
        if not video_ids:
            continue

        for vid in video_ids:
            print(f"📥 Fetching comments for {vid} ({category})")
            logging.info(f"Fetching comments for video {vid} ({category})")

            comments = fetch_comments(vid, category)
            if comments:
                all_comments.extend(comments)
                logging.info(f"✅ Collected {len(comments)} comments from {vid} ({category})")
            else:
                logging.warning(f"⚠️ No comments collected for {vid} ({category})")

    # ------------------------------
    # 🔹 Step 4: Save dataset
    # ------------------------------
    if all_comments:
        df = pd.DataFrame(all_comments)
        os.makedirs("data/raw", exist_ok=True)
        save_path = "data/raw/youtube_bulk_raw.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved {len(df)} comments into {save_path}")
        logging.info(f"✅ Final dataset saved with {len(df)} comments")
    else:
        print("\n⚠️ No comments collected at all.")
        logging.warning("⚠️ No comments collected in this run")
