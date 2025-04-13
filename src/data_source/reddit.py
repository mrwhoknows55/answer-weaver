import praw
from praw.models import MoreComments
from typing import List, Dict

from src.config import settings, logger

try:
    reddit = praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        read_only=True # Set to True if you only need to read data
    )
    logger.info(f"PRAW instance initialized for subreddit: {settings.target_subreddit}")
except Exception as e:
    logger.exception("Failed to initialize PRAW Reddit instance.")
    raise SystemExit("Reddit API connection error, exiting.") from e

def get_comments_text(submission: praw.models.Submission, max_comments: int = 20, max_depth: int = 5) -> str:
    """
    Fetches and concatenates comment text from a submission up to specified limits.
    Handles 'MoreComments' objects.
    """
    comments_text = []
    comment_count = 0
    submission.comments.replace_more(limit=None) # Replace all MoreComments objects

    for comment in submission.comments.list():
        if comment_count >= max_comments:
            break
        if isinstance(comment, MoreComments) or (hasattr(comment, 'depth') and comment.depth >= max_depth):
            continue
        if comment.body and comment.body != '[deleted]' and comment.body != '[removed]':
            comments_text.append(comment.body)
            comment_count += 1

    return "\n---\n".join(comments_text) # Separate comments clearly

def fetch_subreddit_posts(subreddit_name: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Fetches top posts and their comments from a given subreddit.
    """
    posts_data = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        logger.info(f"Fetching top {limit} posts from r/{subreddit_name}...")
        for submission in subreddit.hot(limit=limit):
            if submission.stickied:
                continue

            post_content = f"Title: {submission.title}\n\n{submission.selftext}"
            comments = get_comments_text(submission)

            posts_data.append({
                "id": submission.id,
                "title": submission.title,
                "url": submission.url,
                "content": post_content.strip(),
                "comments": comments,
                "combined_text": f"{post_content.strip()}\n\nComments:\n{comments}" # Text to be embedded
            })
        logger.info(f"Successfully fetched {len(posts_data)} posts.")
    except praw.exceptions.PRAWException as e:
        logger.exception(f"An error occurred while fetching posts from r/{subreddit_name}: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

    return posts_data

if __name__ == '__main__':
    logger.info("Running reddit (fetcher) directly for testing...")
    try:
        from src.config import settings
        fetched_posts = fetch_subreddit_posts(settings.target_subreddit, limit=5)
        if fetched_posts:
            logger.info(f"Fetched {len(fetched_posts)} posts. Example post:")
            # Print limited details to avoid overwhelming output
            example_post = fetched_posts[0]
            print(f"  ID: {example_post['id']}")
            print(f"  Title: {example_post['title']}")
            print(f"  URL: {example_post['url']}")
            print(f"  Content Preview: {example_post['content'][:100]}...")
            print(f"  Comments Preview: {example_post['comments'][:100]}...")
    except Exception as e:
        logger.exception(f"Failed to fetch posts during direct testing: {e}")
    else:
        logger.warning("No posts fetched.")