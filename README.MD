🛒 ProdSense: Product Recommendations Done Differently
ProdSense is a Streamlit app that digs into product URLs and user preferences to deliver recommendations. It scrapes e-commerce sites, hunts for alternatives, pulls community feedback, and finds YouTube reviews, then wraps it up in a Markdown output. Built for tinkerers who want to see what AI agents can do.
Disclaimer: This is a learning project. Expect rough edges—APIs can fail, data can be spotty. It’s an experiment, not a polished tool.

🔍 What ProdSense Does
Takes a product URL (Amazon.in/Flipkart) or preferences, then:

Scrapes product details.
Finds up to three alternative products.
Gathers user sentiments from forums/Reddit.
Locates 1-2 YouTube review videos.
Delivers a Markdown recommendation.


🛠️ The ProdSense Crew: Agents, Tasks, Tools
Agents

URL Identifier: Finds the product URL.
Input Analyst: Parses preferences, scrapes product data.
Market Researcher: Seeks alternative products.
Community Analyst: Collects user feedback.
Video Specialist: Grabs YouTube reviews.
Recommendation Expert: Builds the final recommendation.

Tasks

Identify URL: Picks the right URL.
Parse Input: Extracts needs and product info.
Find Alternatives: Scouts alternatives.
Gather Sentiment: Pulls pros/cons from communities.
Find Videos: Finds review videos.
Synthesize Recommendation: Creates a Markdown output.

Tools

FirecrawlScrapeWebsiteTool: Scrapes product pages.
SerperDevTool: Handles web searches.
ComposioToolSet (Reddit): Fetches Reddit feedback.

Disclaimer: It’s a learning experiment. Results vary based on API reliability and data quality.

📝 Example Usage

Product URL: https://www.amazon.in/dp/B08L5T44CQ
Preferences: "Smartphone under 25k, gaming, good battery."
Hit "Get Recommendations" for a Markdown report.

Recording ProdSense: I’ll be screen-recording a live run. Watch the crew in action—warts and all.

🐛 Troubleshooting

API Issues: Check .env for valid SERPER_API_KEY, GOOGLE_API_KEY, FIRECRAWL_API_KEY, COMPOSIO_API_KEY.
Scraping Fails: Ensure URL is a valid Amazon.in/Flipkart product page.
No Output: Refine preferences or check API limits.


🤔 Why ProdSense?
It’s a raw look at AI agents tackling product research. Not perfect, but a solid starting point for enthusiasts who like to dig into code and data. Expect to tweak and debug.
Crafted as a learning experiment by a seasoned tinkerer.
