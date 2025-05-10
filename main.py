import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai_tools import FirecrawlScrapeWebsiteTool
from composio_crewai import ComposioToolSet


# from langchain_google_genai import ChatGoogleGenerativeAI # Not directly used if crewai.LLM handles it
from crewai import LLM # Use crewai's LLM wrapper

# Set page config and title
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 ProdSense")
st.markdown("Enter a product URL and your preferences to get personalized recommendations.")

# Create a form for user inputs
with st.form("product_recommendation_form"):
    product_url = st.text_input(
        "Product URL (Amazon.in or Flipkart - Optional if URL is in preferences)",
        placeholder="https://www.amazon.in/product-url..."
    )
    
    user_preferences = st.text_area(
        "Your Preferences (Describe your needs, budget, usage. You can also include the product URL here if not provided above)",
        placeholder="Describe your needs, budget, and usage..."
    )
    
    submit_button = st.form_submit_button("Get Recommendations")

# Function to run the product crew
def run_product_crew(product_url, user_preferences):
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for necessary API keys
    serper_api_key = os.getenv("SERPER_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not serper_api_key:
        st.error("SERPER_API_KEY must be set in the .env file.")
        return
    if not google_api_key: # Also check for Google API Key for the LLMs
        st.error("GOOGLE_API_KEY must be set in the .env file.")
        return
    
    # -------------------------
    # Initialize LLMs
    # -------------------------
    # Use smaller model for most tasks to manage rate limits and costs
    llm_fast = LLM(
        model="gemini/gemini-2.0-flash-lite",
        temperature=0.7,
        api_key=os.getenv('GOOGLE_API_KEY')
    ) 
    
    # Use larger model for final synthesis requiring more reasoning
    llm_capable = LLM(
        model="gemini/gemini-2.0-flash-lite",
        temperature=0.7,
        api_key=os.getenv('GOOGLE_API_KEY')
    ) 
    
    # -------------------------
    # Initialize Tools
    # -------------------------
    firecrawl_tool = FirecrawlScrapeWebsiteTool(url=product_url,api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    serper_market_research_tool = SerperDevTool(
    api_key=os.getenv("SERPER_API_KEY"),
    gl='in',  # Keep India-specific for product alternatives
    hl='en',
    n_results=3  # We only need up to 3 alternatives
    )

    serper_community_tool = SerperDevTool(
    api_key=os.getenv("SERPER_API_KEY"),
    gl='in',  # Focus on Indian user feedback
    hl='en',
    n_results=5  # Get a few snippets to gauge sentiment
    )

    serper_video_tool = SerperDevTool(
    api_key=os.getenv("SERPER_API_KEY"),
    gl='in',  # Prioritize Indian reviews if possible
    hl='en',
    n_results=2  # We only need 1-2 credible videos
    )   
    composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"),)


    # Get Composio Reddit Search Tool
    reddit_search_tools = composio_toolset.get_tools(actions=['REDDIT_SEARCH_ACROSS_SUBREDDITS'])
    reddit_search_tool = reddit_search_tools[0] if reddit_search_tools else None

    
    # -------------------------
    # Define Agents
    # -------------------------

    # New Agent: URL Identifier
    url_identifier_agent = Agent(
        role='URL Identifier and Extractor',
        goal=(
            'Accurately determine the primary product URL {user_preferences} {product_url} to be analyzed. '
            'This involves checking a dedicated URL input field and searching within user preference text if the dedicated field is empty or the URL there is invalid.'
        ),
        backstory=(
            'A meticulous agent specializing in identifying the most relevant product URL. '
            'It prioritizes explicit URLs from dedicated fields but is skilled at finding them embedded in text if needed. '
            'It ensures only valid e-commerce product page URLs (e.g. Amazon.in, Flipkart) are selected.'
        ),
        tools=[], # LLM's inherent capabilities are sufficient for this task.
        llm=llm_fast,
        verbose=True,
        allow_delegation=False
    )

    # Agent 1: User Input & Product Analyst
    input_analyst = Agent(
        role='Input Analyst',
        goal='Intelligently parse user preferences {user_preferences} and {product_url}, using the product URL identified by the URL Identifier agent, scrape precise details from that product page.',
        backstory='A tech-savvy detective who decodes user needs and digs deep into product pages to uncover critical details, always working with a confirmed product URL.',
        tools=[firecrawl_tool],
        llm=llm_capable,
        verbose=True,
        allow_delegation=False
    )

    market_researcher = Agent(
        role='Market Researcher',
        goal='Find three alternative products from any e-commerce platform, prioritizing specs that align with the user specific needs.',
        backstory='A gadget enthusiast who scours the web to find the best product matches for your unique preferences.',
        tools=[serper_market_research_tool],
        llm=llm_capable,
        verbose=True,
        allow_delegation=False
    )

    community_analyst = Agent(
        role='Community Analyst',
        goal='Gather authentic user feedback from forums, social media, and review sites, focusing on sentiments relevant to the users needs.',
        backstory='A community whisperer who taps into real-world opinions to reveal what users truly think.',
        tools=[serper_community_tool, reddit_search_tool] if reddit_search_tool  else [serper_community_tool],
        llm=llm_fast,
        verbose=True,
        allow_delegation=False
    )

    video_specialist = Agent(
        role='Video Specialist',
        goal='Source 1-2 credible YouTube review videos tailored to the product and user needs.',
        backstory='A tech vloggers sidekick, finding the most trustworthy video reviews to boost your confidence.',
        tools=[serper_video_tool],
        llm=llm_capable,
        verbose=True,
        allow_delegation=False
    )

    recommendation_expert = Agent(
        role='Recommendation Expert',
        goal='Synthesize all insights into an enthusiastic, tailored recommendation that feels like advice from a tech-buff friend. The final output MUST be a single markdown string.',
        backstory='Your go-to tech guru, delivering exciting and spot-on product picks with a personal touch.',
        tools=[],
        llm=llm_capable,
        verbose=True,
        allow_delegation=False
    )

    # -------------------------
    # Define Tasks
    # -------------------------

    # New Task: Identify and Finalize Product URL
    task_identify_url = Task(
        description=(
            "Your primary goal is to determine the single, definitive product URL to be scraped and analyzed. "
            "You will receive two pieces of information:\n"
            "1. A 'product_url' potentially from a dedicated input form: '{product_url}'\n"
            "2. A 'user_preferences' text which might also contain a URL: '{user_preferences}'\n\n"
            "Follow these steps to determine the 'final_product_url':\n"
            "a. First, check the '{product_url}'. If it is a valid, complete URL for a product page on a major e-commerce site (e.g., Amazon.in, Flipkart), then this is your 'final_product_url'.\n"
            "b. If '{product_url}' is empty, not a URL, or not a valid e-commerce product page URL, then thoroughly scan the '{user_preferences}' text. Look for any e-commerce product URLs (Amazon.in, Flipkart primarily). If you find one, that becomes your 'final_product_url'. If multiple are found in preferences, choose the most prominent or first one that seems to be the main product of interest.\n"
            "c. If a valid URL is found in '{product_url}', it takes precedence even if a URL is also in '{user_preferences}'.\n"
            "d. If no valid e-commerce product URL can be found in either '{product_url}' or '{user_preferences}', then the 'final_product_url' should be the string 'None'.\n"
            "e. Ensure the extracted URL is clean and directly usable (e.g., remove unnecessary tracking parameters if easily identifiable, but prioritize getting the base product URL correct)."
        ),
        expected_output=(
            "A dictionary containing a single key 'final_product_url'. "
            "The value for this key should be the determined product URL string (e.g., 'https://www.amazon.in/some-product-link/...'). "
            "If no valid URL is found, the value should be the literal string 'None'.\n"
            "Example for success: {{'final_product_url': 'https://www.amazon.in/dp/B0801Y8R32'}}\n"
            "Example for failure: {{'final_product_url': 'None'}}"
        ),
        agent=url_identifier_agent
    )

    task_parse_input = Task(
    description=(
        "The preceding 'task_identify_url' has attempted to find a 'final_product_url' which is now in your context. You also have the original '{user_preferences}' available.\n"
        "Your primary goals are to understand the user's intent and extract key information, regardless of whether a specific product URL was found:\n"
        "1. **Analyze User Preferences ('{user_preferences}'):**\n"
        "   - **Identify 'product_type'**: Determine the general category of product the user is interested in (e.g., 'smartphone', 'laptop', 'book', 'coffee maker'). Be as specific as possible based on the text.\n"
        "   - **Extract 'budget'**: Look for any explicit or implicit mention of a budget in INR. Extract the numeric value (e.g., 'around 15000' -> 15000, 'under 10k' -> 10000). If no budget is mentioned, set to None.\n"
        "   - **Identify 'specific_needs'**: List the key features, functionalities, or requirements the user has mentioned (e.g., ['good battery life', 'for gaming', 'portable', 'with a large display']).\n"
        "   - **Identify potential product URLs**: Scan the '{user_preferences}' again for any valid-looking URLs from major e-commerce sites (Amazon, Flipkart, etc.). If found, note them, but the 'final_product_url' from the previous task takes precedence for direct scraping.\n"
        "2. **Process 'final_product_url' from Context:**\n"
        "   - If 'final_product_url' is a valid URL (not 'None'):\n"
        "     - Use FirecrawlCrawlWebsiteTool to scrape the product page.\n"
        "     - Extract: precise 'product_name', 'price' (INR), 'key_specs' relevant to identified 'specific_needs' or general product type, 'key_features', and 1-2 concise 'reviews' or ratings.\n"
        "     - If scraping fails, set 'original_product' details to indicate failure (e.g., name: 'Scraping Failed', url: 'final_product_url', price: None, error: '<reason>').\n"
        "   - If 'final_product_url' is 'None': Set 'original_product' to indicate no URL (e.g., name: 'No Product URL Provided', url: 'None', price: None, error: 'No valid product URL was found for analysis. Recommendations will be based on user preferences only.').\n"
        "3. **Structure Output:** Return a single JSON-compatible dictionary with keys: 'user_context' (containing product_type, budget, specific_needs) and 'original_product' (containing scraped details or status).\n"
    ),
    expected_output=(
        "A dictionary with two main keys: 'user_context' and 'original_product'.\n"
        "'user_context': {{'product_type': '<extracted_type_or_None>', 'budget': <extracted_budget_INR_or_None>, 'specific_needs': ['<need1>', ...]}}.\n"
        "'original_product': {{'name': '<scraped_name_or_status>', 'price': <price_or_None>, 'key_specs': ['<spec1>', ...], 'key_features': ['<feature1>', ...], 'reviews': ['<review1>', ...], 'url': '<used_url_or_None>', 'error': '<error_if_any_or_None>'}}"
    ),
    agent=input_analyst,
    context=[task_identify_url]
    )

    task_find_alternatives = Task(
    description=(
        "Based on the 'user_context' (especially 'product_type', 'budget', and 'specific_needs') extracted from 'task_parse_input', and the details of the 'original_product' (if a URL was successfully processed):\n"
        "1. **Formulate Targeted Search Queries:** Use SerperDevTool to find up to three alternative products. Prioritize queries that include the 'product_type' and as many relevant 'specific_needs' as possible. Incorporate the 'budget' if available (e.g., 'best <product_type> for <need1> <need2> under <budget> INR'). If 'original_product' details are available (especially price), consider searching within a reasonable price range of that too.\n"
        "2. **Search Globally (with Indian Preference):** Perform the search with a focus on identifying products available in India (`gl='in'`). However, if very few or no relevant Indian alternatives are found, broaden the search slightly while still prioritizing reputable e-commerce platforms.\n"
        "3. **Scrape Alternatives:** For each promising alternative found in the search results:\n"
        "   - Use FirecrawlCrawlWebsiteTool to get its 'name', 'price' (INR), one key 'spec' that strongly aligns with the user's 'specific_needs' or is a major selling point, and its 'url'.\n"
        "   - Attempt to extract 1-2 brief 'review' snippets from the product page if easily accessible.\n"
        "4. **Filter and Validate:** Prioritize alternatives that closely match the 'product_type' and 'specific_needs'. If a 'budget' was specified, aim for products within a reasonable range. Avoid suggesting the 'original_product' if it was successfully identified.\n"
        "5. **Handle No Alternatives:** If fewer than three suitable alternatives are found, return the ones identified. If no relevant alternatives meeting the user's criteria can be found, return an empty list and a clear 'warning' message explaining why (e.g., 'No suitable alternatives found within the specified budget and needs for a <product_type>.').\n"
    ),
    expected_output=(
        "A list of dictionaries, where each dictionary represents an alternative product:\n"
        "[{'name': '<name>', 'price': <int_or_str_unavailable>, 'key_spec': '<relevant_spec>', 'url': '<url>', 'reviews': ['<review1>', ...], 'warning': '<warning_message_or_None>'}, ...]\n"
        "An empty list `[]` and a top-level warning message if no suitable alternatives are found."
    ),
    agent=market_researcher,
    context=[task_parse_input]
    )

    task_gather_sentiment = Task(
    description=(
        "Using the 'product_name' and 'product_type' (from 'user_context' in 'task_parse_input') for both the 'original_product' (if available) and the list of 'alternatives' (from 'task_find_alternatives'), as well as the user's 'specific_needs':\n"
        "1. **Formulate Sentiment Search Queries:** For each product (original and alternatives), generate targeted search queries using SerperDevTool to find user feedback. Examples:\n"
        "   - '<product_name> reviews and opinions'\n"
        "   - '<product_type> user forum discussion <specific_need_keyword>'\n"
        "   - '<product_name> problems and issues'\n"
        "   - 'is <product_name> worth buying India?'\n"
        "2. **Focus on Diverse Sources:** Search for feedback on forums, social media (excluding direct e-commerce product pages), independent review sites, and tech communities. Prioritize real user experiences and opinions.\n"
        "3. **Extract Pros and Cons:** For each product, identify 1-2 distinct pros and 1-2 distinct cons based on the gathered feedback. Focus on aspects relevant to the 'product_type' and the user's 'specific_needs'.\n"
        "4. **Handle Scarcity of Feedback:** If specific user feedback for a product is limited or unavailable, clearly indicate this with a 'warning' message for that product's sentiment entry (e.g., 'Limited user reviews found for <product_name>.').\n"
    ),
    expected_output=(
        "A dictionary with two keys: 'original_sentiment' and 'alternatives_sentiment'.\n"
        "'original_sentiment': {{'product_name': '<name_or_NA>', 'pros': ['<pro1>'], 'cons': ['<con1>'], 'warning': '<warning_or_None>'}} (Set to NA or empty if no original product was analyzed).\n"
        "'alternatives_sentiment': A list of dictionaries, each for an alternative:\n"
        "[{{'name': '<alt_name>', 'pros': ['<pro1>'], 'cons': ['<con1>'], 'warning': '<warning_or_None>'}}, ...]"
    ),
    agent=community_analyst,
    context=[task_parse_input, task_find_alternatives]
    )

    task_find_videos = Task(
    description=(
        "Using the 'product_name' and 'product_type' (from 'user_context' in 'task_parse_input') of the 'original_product' (if available), and considering the user's 'specific_needs':\n"
        "1. **Check for Original Product Details:** If the 'original_product_name' from 'task_parse_input' is not valid (e.g., 'No Product URL Provided'), you cannot search for specific videos. Output an empty list with a warning.\n"
        "2. **Formulate Targeted YouTube Search Queries:** If a valid product name exists, use SerperDevTool to search YouTube for relevant review videos. Examples:\n"
        "   - '<original_product_name> review India'\n"
        "   - 'best <product_type> <specific_need_keyword> review'\n"
        "   - '<original_product_name> vs alternatives'\n"
        "   - 'top <product_type> in 2024 India'\n"
        "3. **Prioritize Credibility and Relevance:** Aim to find 1-2 recent (ideally within the last 2 years) and credible review videos, preferably from Indian tech reviewers if available. Focus on videos that discuss aspects relevant to the user's 'specific_needs'.\n"
        "4. **Handle No Relevant Videos:** If no suitable videos are found after a thorough search, return an empty list with a clear warning (e.g., 'No specific or recent video reviews found for <original_product_name>.').\n"
    ),
    expected_output=(
        "A dictionary containing video URLs and a warning if applicable:\n"
        "{{'video_urls': ['<youtube_url1>', '<youtube_url2>'], 'warning': '<warning_or_None_if_videos_found_or_product_invalid>'}}"
    ),
    agent=video_specialist,
    context=[task_parse_input]
    )

    task_synthesize_recommendation = Task(
    description=(
        "Synthesize ALL gathered information to provide a comprehensive and helpful product recommendation:\n"
        "- 'user_context' and 'original_product' details (including any errors) from 'task_parse_input'.\n"
        "- Alternative products from 'task_find_alternatives'.\n"
        "- Community sentiment for original and alternatives from 'task_gather_sentiment'.\n"
        "- Video reviews for the original product (if available) from 'task_find_videos'.\n"
        "Adopt the persona of a friendly and knowledgeable tech buddy offering advice.\n"
        "The final output MUST be a single, well-formatted Markdown string.\n"
        "Structure the Markdown content logically:\n"
        "1. **Hey there! Let's Find You the Perfect Product!** Enthusiastic opening, acknowledging the user's request.\n"
        "2. **Your Initial Thoughts (If a Product URL Was Provided):** If 'final_product_url' was valid, briefly discuss the original product, its key features, and how it aligns (or doesn't) with the user's needs. If there was an error scraping, clearly state that.\n"
        "3. **What Others Are Saying:** Present the pros and cons gathered from the community for the original product (if analyzed) and each alternative. Highlight any warnings about limited feedback.\n"
        "4. **Exploring Alternatives:** Introduce the alternative products found, highlighting their key features and prices. Explain why they might be suitable based on the user's 'product_type' and 'specific_needs'.\n"
        "5. **My Top Pick:** Based on all the information, provide a clear recommendation. Justify your choice by referencing the user's needs, budget, features, community sentiment, and any video insights. If no clear recommendation can be made due to lack of information, explain this honestly.\n"
        "6. **Dive Deeper (Video Reviews if Available):** If video reviews were found for the original product, include links to them.\n"
        "Use Markdown for clear formatting (headings, bold, italics, lists, links). Handle missing data gracefully (e.g., 'Not available', 'No specific feedback found')."
    ),
    expected_output=(
        "A single, complete Markdown string containing the full product recommendation, formatted for direct display to the user."
    ),
    agent=recommendation_expert,
    context=[task_parse_input, task_find_alternatives, task_gather_sentiment, task_find_videos]
    )

    # Create Crew
    product_crew = Crew(
        agents=[
            url_identifier_agent, # New agent added first
            input_analyst,
            market_researcher,
            community_analyst,
            video_specialist,
            recommendation_expert
        ],
        tasks=[
            task_identify_url,    # New task runs first
            task_parse_input,
            task_find_alternatives,
            task_gather_sentiment,
            task_find_videos,
            task_synthesize_recommendation
        ],
        process=Process.sequential,
        verbose=True, # Increased verbosity for debugging if needed (0, 1, or 2)
        manager_llm=llm_capable # Using a capable LLM for orchestration
    )
    
    with st.spinner("🤖 Agents are brewing your personalized recommendations... This might take a moment (or a few)!"):
        try:
            # Inputs for the kickoff should match what the first task(s) expect.
            # task_identify_url expects 'product_url' and 'user_preferences'.
            # task_parse_input also expects 'user_preferences' directly.
            result = product_crew.kickoff(inputs={
                'product_url': product_url,
                'user_preferences': user_preferences
            })
            
            # CrewAI's kickoff typically returns the output of the last task.
            # If using newer versions of CrewAI that return a CrewOutput object:
            if hasattr(result, 'raw') and result.raw is not None:
                return str(result.raw) # .raw usually holds the direct output of the last task
            elif isinstance(result, str): # If result is already a string
                return result
            else:
                # Fallback or for older versions if result is the tasks's output directly
                st.warning(f"Unexpected result format from crew.kickoff(): {type(result)}. Attempting to stringify.")
                return str(result)

        except Exception as e:
            import traceback
            st.error(f"An error occurred while the agents were working: {e}")
            st.error(f"Traceback: {traceback.format_exc()}") # More detailed error for debugging
            return None


# When the form is submitted
if submit_button:
    if not user_preferences: # product_url is now optional if preferences has it
        st.error("Please describe your preferences. You can include a product URL there too.")
    # Removed check for product_url as it's handled by the agent now
    # else if not product_url and not any(url_pattern_in user_preferences): # more complex check not needed here
    # st.error("Please provide a product URL either in its field or within your preferences.")
    else:
        progress_placeholder = st.empty()
        progress_placeholder.info("🚀 Launching the ProdSense crew... Your tech buddies are on the case!")
        
        result_markdown = run_product_crew(product_url, user_preferences)
        
        if result_markdown:
            progress_placeholder.success("✅ Your Personalized Product Insights are Ready!")
            st.markdown(result_markdown)
        else:
            progress_placeholder.error("⚠️ Something went wrong, and we couldn't generate recommendations. Please check the error messages or try again.")