import requests

GOOGLE_FACT_CHECK_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def fact_check_news(news_text):
    """
    Queries Google Fact Check API to verify the news claim.
    
    Args:
        news_text (str): The news statement to fact-check.

    Returns:
        dict: Fact check response including rating and source.
    """
    params = {
        "query": news_text,
        "key": GOOGLE_FACT_CHECK_API_KEY,
    }

    response = requests.get(GOOGLE_FACT_CHECK_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if "claims" in data:
            claim = data["claims"][0]  # Get the top claim
            rating = claim["claimReview"][0]["textualRating"]
            publisher = claim["claimReview"][0]["publisher"]["name"]
            return {"rating": rating, "source": publisher}
    
    return {"rating": "Unknown", "source": "No fact-checking source found"}
