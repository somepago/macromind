import yfinance as yf
from openai import OpenAI
import datetime
import os
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from constants import current_news_data, current_stock_data
from typing import List
# Setup your OpenAI API Key
openai_key = "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"  # Or paste directly: "sk-..."

client = OpenAI(api_key=openai_key)


# Define stock or ETF to analyze
stock_symbol = "GLD"  # You can change to any from your ETF/stock list
class CommodityPricePredictor:
    def __init__(self, news_df=None, price_df=None, num_days=2) -> None:
        if news_df is not None:
            self.news_df = pd.read_csv(news_df)
        else:
            self.fetch_news(num_days)
        if price_df is not None:
            self.commodity_price_df = pd.read_csv(price_df)
        else:
            self.fetch_price(12)

    def fetch_news(self,num_days):
        pass # TODO

    def fetch_price(self,num_days):
        pass # TODO

    @staticmethod
    def get_recent_averages(df, commodity, start_date, num_days=10):
        from datetime import datetime
        
        # Parse the start_date string to datetime
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Filter for the specific commodity
        filtered = df[df['Commodity'].str.lower() == commodity.lower()].copy()
        
        # Ensure 'Date' column is in datetime format
        filtered['Date'] = pd.to_datetime(filtered['Date'])
        
        # Filter to only rows before the start_date
        filtered = filtered[filtered['Date'] < start_date]
        
        # Group by date to get daily average
        daily_avg = filtered.groupby(['Date', 'Commodity'], as_index=False)['Average'].mean()
        
        # Sort by date descending and get the last `num_days` rows
        recent = daily_avg.sort_values(by='Date', ascending=False).head(num_days)
        
        # Reverse rows and format lines
        lines = [
            f"{row['Date'].strftime('%Y-%m-%d')}: Average Price ${row['Average']:.2f}"
            for _, row in list(recent.iterrows())[::-1]  # convert to list, then reverse
        ]
        
        return "\n".join(lines)

    def format_stock_data(self, commodity,startdate, num_days=10):
        return self.get_recent_averages(self.commodity_price_df, commodity,startdate, num_days)

    def format_news(self, commodity, start_date, num_headlines=10):
        df = self.news_df.copy()
        
        # Ensure datetime is in datetime format
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        
        # Filter by commodity and start_date
        df = df[(df['commodity'] == commodity) & (df['datetime'] >= pd.to_datetime(start_date))]
        
        # Sort latest first
        df = df.sort_values(by='datetime', ascending=False)
        
        # Drop rows with NaN in headline or summary
        df = df.dropna(subset=['headline'])
        
        # Combine headline and summary
        combined = df.apply(
            lambda row: f"{row['headline'].strip()}: {row['summary'].strip()}"
            if pd.notnull(row['summary']) else row['headline'].strip(),
            axis=1
        )

        # Filter out any remaining NaN values
        combined = combined.dropna()
        
        return combined.values.tolist()[:num_headlines]

    # Step 4: Build the prompt
    @staticmethod
    def build_prompt(commodity, stock_text, headlines,assistant_mode="verbose"):
        headline_text = "\n- ".join(headlines)
        if assistant_mode != "verbose":
            prompt = f"""
            You are a financial analyst.

            Here is the average commodity price data for {commodity} for the last 10 trading days:
            {stock_text}

            Here are some recent news headlines relevant to this commodity:
            - {headline_text}

            Based on the above, do you expect the price of {commodity} to go UP, DOWN, or STAY THE SAME today compared to the previous trading day?

            Reply with exactly one word: UP, DOWN, or SAME.
            """
        else:
            prompt = f"""
            You are a financial analyst.

            Here is the average commodity price data for {commodity} for the last 10 trading days:
            {stock_text}

            Here are some recent news headlines relevant to this commodity:
            - {headline_text}

            Based on the above, do you expect the price of {commodity} to go UP, DOWN, or STAY THE SAME today compared to the previous trading day?

            Reply with exactly one word: UP, DOWN, or SAME. Also give a short explanation of this prediction.
            """
        return prompt.strip()


    # Step 5: Query the LLM (GPT-4)
    @staticmethod
    def predict_direction(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def __call__(self,
            commodity: str,
            startdate: str,
            assistant_mode: str = "verbose"):
        stock_prompt = self.format_stock_data(commodity,startdate)
        headlines = self.format_news(commodity,startdate)
        final_prompt = self.build_prompt(commodity, stock_prompt, headlines,assistant_mode) 
        final_prediction = self.predict_direction(final_prompt)
        pred, _ ,explanation = final_prediction.split("\n")
        return pred, explanation
    

    def pred_all(self,
                 commodities: List[str],
                 startdate: str,
                 assistant_mode: str = "verbose"):
        preds, explanations = [], []
        for commodity in commodities:
            stock_prompt = self.format_stock_data(commodity,startdate)
            headlines = self.format_news(commodity,startdate)
            final_prompt = self.build_prompt(commodity, stock_prompt, headlines,assistant_mode) 
            final_prediction = self.predict_direction(final_prompt)
            pred, _ ,explanation = final_prediction.split("\n")
            preds.append(pred)
            explanations.append(explanation)

        return preds, explanations


        




if __name__ == "__main__":
    com_pred = CommodityPricePredictor(news_df=current_news_data,price_df=current_stock_data)
    commodity = "coffee"
    startdate = "2025-04-06"
    future_direction, explanation = com_pred(commodity,startdate,"verbose")
    print(future_direction, explanation)



    # # Initialize predictor
    # com_pred = CommodityPricePredictor(
    #     news_df="data_prep/scraped_news_finhub/commodity_news_20250410_195943.csv",
    #     price_df="data_prep/commodity_data_60days.csv"
    # )

    # commodity = "coffee"
    # start_date = datetime.strptime("2025-02-04", "%Y-%m-%d")
    # end_date = datetime.strptime("2025-03-05", "%Y-%m-%d")

    # date_cursor = start_date
    # results = []

    # while date_cursor <= end_date:
    #     try:
    #         pred = com_pred(commodity, date_cursor.strftime("%Y-%m-%d"))
    #         results.append({
    #             "date": date_cursor.strftime("%Y-%m-%d"),
    #             "prediction": pred
    #         })
    #     except Exception as e:
    #         print(f"Failed for {date_cursor.strftime('%Y-%m-%d')}: {e}")
    #     date_cursor += timedelta(days=1)

    # # Convert to DataFrame
    # results_df = pd.DataFrame(results)

    # # Optional: Plot "up" vs "down" signals over time
    # results_df['label'] = results_df['prediction'].str.lower().map({
    #     'up': 1, 'down': -1
    # })
    # results_df['date'] = pd.to_datetime(results_df['date'])

    # plt.figure(figsize=(12, 5))
    # plt.plot(results_df['date'], results_df['label'], marker='o', linestyle='-')
    # plt.title(f"{commodity.title()} price movement prediction")
    # plt.ylabel("Prediction (1=up, -1=down)")
    # plt.xlabel("Date")
    # plt.grid(True)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(f"plots/{commodity}_price_direction_prediction.png")
    # plt.close()
