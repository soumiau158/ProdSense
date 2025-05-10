# ProdSense: Multi-Agent Product Recommendation System

ProdSense is a Streamlit app powered by CrewAI that scrapes product details from any product website, hunts for alternatives, pulls community sentiments, and finds YouTube reviews. It delivers a Markdown recommendation tailored to your needs. Built for tinkerers who want to dive into AI-driven product research.

**Disclaimer**: This is an experimental learning project. APIs can be flaky, and data quality varies. Expect to debug and tweak.

## Features

- Scrapes product specs from any product website (optimized for e-commerce).
- Finds up to three alternative products based on your preferences.
- Analyzes user feedback from forums and Reddit.
- Curates 1-2 relevant YouTube review videos.
- Outputs a concise Markdown recommendation.

## The ProdSense Crew

### Agents
- **URL Identifier**: Detects the product URL from inputs.
- **Input Analyst**: Parses preferences and scrapes product data.
- **Market Researcher**: Discovers alternative products.
- **Community Analyst**: Gathers user sentiments.
- **Video Specialist**: Finds YouTube review videos.
- **Recommendation Expert**: Crafts the final Markdown output.

### Tasks
- **Identify URL**: Locates the target product URL.
- **Parse Input**: Extracts user needs and product details.
- **Find Alternatives**: Scouts alternative products.
- **Gather Sentiment**: Collects community pros and cons.
- **Find Videos**: Retrieves relevant review videos.
- **Synthesize Recommendation**: Builds the Markdown recommendation.

### Tools
- **FirecrawlScrapeWebsiteTool**: Scrapes product pages.
- **SerperDevTool**: Powers web searches for alternatives, sentiments, and videos.
- **ComposioToolSet (Reddit)**: Fetches Reddit feedback.

**Disclaimer**: Results depend on API reliability and data availability. It’s a learning experiment, not a production tool.

## Example Usage

- **Product URL**: `https://www.example.com/product`
- **Preferences**: "Smartphone under 25k, gaming, long battery life."
- Click "Get Recommendations" for a Markdown report.
- [Watch ProdSense in Action](assets/prodsense-demo.mp4)  
  *Screen-recorded demo showing the app’s flow—raw and unpolished.*

## Troubleshooting

- **API Errors**: Verify `SERPER_API_KEY`, `GOOGLE_API_KEY`, `FIRECRAWL_API_KEY`, and `COMPOSIO_API_KEY` in `.env`.
- **Scraping Issues**: Ensure the URL points to a valid product page.
- **No Results**: Refine preferences or check API quotas.

## Why ProdSense?

ProdSense is a playground for exploring multi-agent AI, web scraping, and recommendation systems. It’s rough around the edges but offers a solid starting point for enthusiasts who enjoy dissecting code and experimenting with data.
