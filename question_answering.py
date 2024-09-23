import pandas as pd
from transformers import pipeline

# Load dataset
df = pd.read_csv('marketing_campaign_dataset_small.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Clean the Acquisition_Cost column
df['Acquisition_Cost'] = df['Acquisition_Cost'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Initialize the question-answering pipeline
nlp = pipeline('question-answering', model='mrm8488/bert-tiny-5-finetuned-squadv2')


# Function to generate the context from the dataset
def generate_context(df):
    """
    Generate a context string from the dataframe for answering questions.
    """
    context = "\n".join(
        f"Campaign ID {row['Campaign_ID']} by {row['Company']} is a {row['Campaign_Type']} campaign "
        f"targeted at {row['Target_Audience']} for {row['Duration']} days. The campaign was run through {row['Channel_Used']} "
        f"and had a conversion rate of {row['Conversion_Rate']}. The acquisition cost was ${row['Acquisition_Cost']}, "
        f"resulting in an ROI of {row['ROI']}. The campaign was conducted in {row['Location']} and the language used was {row['Language']}. "
        f"It received {row['Clicks']} clicks and {row['Impressions']} impressions, with an engagement score of {row['Engagement_Score']}. "
        f"The target customer segment was {row['Customer_Segment']} and the campaign launched on {row['Date']}."
        for _, row in df.iterrows()
    )
    return context


# Function to answer user questions using the model
def extract_answer(question, context, chunk_size=500):
    """
    Extract the best answer for a question using a pre-trained NLP model.

    Args:
        question (str): The user's question.
        context (str): The context to search for the answer.
        chunk_size (int): Size of context chunks for processing.

    Returns:
        str: The best answer found in the context.
    """
    context_chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
    best_answer, highest_score = "", 0

    for chunk in context_chunks:
        result = nlp(question=question, context=chunk)
        if result['score'] > highest_score:
            best_answer, highest_score = result['answer'], result['score']

    return best_answer or "Sorry, I couldn't find an answer."


# Generate context from the dataset
context = generate_context(df)


# Function to display campaign statistics
def campaign_statistics():
    """
    Display basic statistics about the campaigns.
    """
    avg_acquisition_cost = df['Acquisition_Cost'].mean()
    avg_roi = df['ROI'].mean()
    max_roi_campaign = df.loc[df['ROI'].idxmax()]

    print(f"\nAverage Acquisition Cost: ${avg_acquisition_cost:.2f}")
    print(f"Average ROI: {avg_roi:.2f}")
    print(
        f"Campaign with the highest ROI: {max_roi_campaign['Campaign_ID']} "
        f"with an ROI of {max_roi_campaign['ROI']}\n"
    )


# Function to filter campaigns based on a target criterion
def filter_campaigns():
    """
    Filter campaigns based on user-specified criteria.
    """
    print("\nFilter options:")
    print("1. Campaign Type")
    print("2. Target Audience")
    print("3. Location")
    print("4. Channel Used")

    option = input("Choose a filter option (1-4): ")

    filters = {
        '1': 'Campaign_Type',
        '2': 'Target_Audience',
        '3': 'Location',
        '4': 'Channel_Used'
    }

    if option in filters:
        filter_value = input(f"Enter the {filters[option].replace('_', ' ').lower()}: ")
        filtered = df[df[filters[option]] == filter_value]
        print(
            f"\nFiltered campaigns:\n"
            f"{filtered[['Campaign_ID', 'Company', 'Campaign_Type', 'Location', 'Target_Audience']]}"
        )
    else:
        print("Invalid option. Returning to main menu.")


# Function to summarize data
def campaign_summary():
    """
    Print a summary of the marketing campaign data.
    """
    print("\nSummary of Marketing Campaign Data:")
    print(df.describe(include='all'))


# Main function to run the chatbot
def main():
    """
    Main function to run the marketing campaign chatbot.
    """
    print("Welcome to the Real-Time Marketing Campaign Chatbot!")
    print("You can ask any question about the marketing campaigns or choose an action.")
    print("Available actions: 'stats', 'filter', 'summary', 'exit'.")

    while True:
        user_input = input("\nAsk a question or choose an action: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'stats':
            campaign_statistics()
        elif user_input.lower() == 'filter':
            filter_campaigns()
        elif user_input.lower() == 'summary':
            campaign_summary()
        else:
            response = extract_answer(user_input, context)
            print(f"\nResponse: {response}")


if __name__ == '__main__':
    main()
