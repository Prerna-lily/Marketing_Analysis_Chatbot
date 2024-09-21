import pandas as pd
from transformers import pipeline
import spacy
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings related to transformers
warnings.filterwarnings("ignore", message="The `clean_up_tokenization_spaces`")

# Load Spacy NLP model for question filtering
nlp_spacy = spacy.load("en_core_web_sm")

# Load dataset and preprocess it
df = pd.read_csv('marketing_campaign_dataset_first2K.csv')
df.columns = df.columns.str.strip()
df['Acquisition_Cost'] = df['Acquisition_Cost'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Initialize the question-answering pipeline with DistilBERT
nlp = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', clean_up_tokenization_spaces=True)

# Caching previous responses
response_cache = {}


# Function to generate focused context for the question
# (This function can be improved if needed)
def generate_context(filtered_df):
    filtered_df = filtered_df.head(10)
    context = "\n".join(
        f"Campaign ID {row['Campaign_ID']} by {row['Company']} is a {row['Campaign_Type']} campaign "
        f"targeted at {row['Target_Audience']} for {row['Duration']} days. It was conducted in {row['Location']} using {row['Channel_Used']}. "
        f"The campaign had a conversion rate of {row['Conversion_Rate']}, an acquisition cost of ${row['Acquisition_Cost']:.2f}, "
        f"and a ROI of {row['ROI']:.2f}."
        for _, row in filtered_df.iterrows()
    )
    return context

# Function to visualize campaign trends
def visualize_trends():
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Date', y='ROI', marker='o')
    plt.title('ROI Trend Over Time')
    plt.xticks(rotation=45)
    plt.ylabel('ROI')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

# Function to summarize general trends in the data
def summarize_trends():
    trend_summary = []
    average_roi = df['ROI'].mean()
    average_acquisition_cost = df['Acquisition_Cost'].mean()
    campaign_types = df['Campaign_Type'].value_counts()

    trend_summary.append(f"The average ROI across campaigns is {average_roi:.2f}.")
    trend_summary.append(f"The average acquisition cost is ${average_acquisition_cost:.2f}.")
    trend_summary.append("Campaign Types Breakdown:")

    for campaign_type, count in campaign_types.items():
        trend_summary.append(f" - {campaign_type}: {count} campaigns")

    return "\n".join(trend_summary)

# Function to provide insights on the effectiveness of different campaign types
def campaign_effectiveness_insights():
    effectiveness_summary = []
    for campaign_type in df['Campaign_Type'].unique():
        subset = df[df['Campaign_Type'] == campaign_type]
        avg_roi = subset['ROI'].mean()
        avg_conversion_rate = subset['Conversion_Rate'].mean()
        effectiveness_summary.append(
            f"{campaign_type}: Average ROI is {avg_roi:.2f}, Average Conversion Rate is {avg_conversion_rate:.2f}."
        )
    return "\n".join(effectiveness_summary)

# Function to extract an answer based on the question
def extract_answer(question):
    if question in response_cache:
        return response_cache[question]

    # Handle specific questions about languages and target audiences
    # ... (previous code) ...

    # Handle general trend questions
    # ... (previous code) ...

    # **New Logic to answer more complex questions**
    if "influencer campaign" in question and "highest conversion rate" in question:
        # Filter influencer campaigns
        influencer_campaigns = df[df['Campaign_Type'] == 'Influencer Marketing Campaign']
        # Find the campaign with the highest conversion rate
        # **Correctly find the maximum conversion rate in the filtered dataframe**
        highest_conversion_rate = influencer_campaigns['Conversion_Rate'].max()
        # **Filter for campaigns with the maximum conversion rate**
        highest_conversion_campaign = influencer_campaigns[influencer_campaigns['Conversion_Rate'] == highest_conversion_rate]
        # **Check if the dataframe is empty**
        if not highest_conversion_campaign.empty:
            # Get the channel used for that campaign
            channel_used = highest_conversion_campaign['Channel_Used'].iloc[0]
            return f"The channel used for the influencer campaign with the highest conversion rate was {channel_used}."
        else:
            return "There are no influencer campaigns with a conversion rate in the dataset."

    # Existing conditions for other questions
    filtered_context = generate_context(df)
    result = nlp(question=question, context=filtered_context)

    if result['score'] > 0.1:
        response_cache[question] = result['answer']
    else:
        response_cache[
            question] = "That's an interesting question! Have you considered how changing the campaign type affects the ROI?"

    return response_cache[question]

# Function to handle "stats" action
def handle_stats():
    average_roi = df['ROI'].mean()
    average_acquisition_cost = df['Acquisition_Cost'].mean()
    print(f"Average ROI: {average_roi:.2f}")
    print(f"Average Acquisition Cost: ${average_acquisition_cost:.2f}")

# Function to handle "filter" action
def handle_filter():
    filter_criteria = input("Enter filtering criteria (e.g., 'Campaign_Type == 'Social Media Campaign''): ")
    try:
        filtered_df = df.query(filter_criteria)
        print("Filtered data:")
        print(filtered_df)
    except:
        print("Invalid filtering criteria.")

# Function to handle "summary" action
def handle_summary():
    print(summarize_trends())

# Function to handle "trends" action
def handle_trends():
    visualize_trends()

# Function to handle "sentiment" action
def handle_sentiment():
    from textblob import TextBlob
    sentiment_results = []
    for _, row in df.iterrows():
        blob = TextBlob(row['Campaign_Description'])
        sentiment_results.append(
            f"Campaign ID {row['Campaign_ID']}: Sentiment - Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}"
        )
    print("Sentiment Analysis:")
    print("\n".join(sentiment_results))

# Main function to run the chatbot
def main():
    print("Welcome to the Real-Time Marketing Campaign Chatbot!")
    print("You can ask any question about the marketing campaigns or choose an action.")
    print("Available actions: 'stats', 'filter', 'summary', 'trends', 'sentiment', 'exit'.")

    while True:
        user_input = input("\nAsk a question or choose an action: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'stats':
            handle_stats()
        elif user_input.lower() == 'filter':
            handle_filter()
        elif user_input.lower() == 'summary':
            handle_summary()
        elif user_input.lower() == 'trends':
            handle_trends()
        elif user_input.lower() == 'sentiment':
            handle_sentiment()
        else:
            response = extract_answer(user_input)
            print(f"\nResponse: {response}")


if __name__ == '__main__':
    main()