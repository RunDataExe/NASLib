import hashlib
import datetime
import requests
import json
import os
import argparse  # Added for command-line argument parsing

# --- Alpha Vantage API Configuration ---
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
COMMITMENT_FILE = "naslib/optimizers/oneshot/gsparsity/seed_commitment.json"
RESULTS_FILE = "naslib/optimizers/oneshot/gsparsity/nas_experiment_seeds.json"


def fetch_stock_price(api_key, stock_ticker, date_str):
    """Fetches the closing stock price for a given ticker and date using Alpha Vantage."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": stock_ticker,
        "apikey": api_key,
        "outputsize": "compact",  # 'compact' for last 100 days, 'full' for more
    }
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if "Error Message" in data:
            print(f"API Error for {stock_ticker}: {data['Error Message']}")
            return None
        if "Note" in data and "API call frequency" in data["Note"]:
            print(f"API Note: {data['Note']}")
            # This might indicate a rate limit issue, but data might still be there
            # or it might be a general warning. Proceed with caution.

        daily_data = data.get("Time Series (Daily)")
        if not daily_data:
            print(
                f"No 'Time Series (Daily)' data found for {stock_ticker}. Response: {data}"
            )
            return None

        if date_str in daily_data:
            return daily_data[date_str]["4. close"]
        else:
            print(
                f"No data found for {stock_ticker} on {date_str}. Available dates might be different (e.g., market closed)."
            )
            print(
                f"Available data keys (dates): {list(daily_data.keys())[:5]}..."
            )  # Show a few available dates
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except KeyError:
        print(
            f"Could not parse stock price from API response for {stock_ticker} on {date_str}."
        )
        print(
            f"Response structure might have changed or data is missing. Full response: {data}"
        )
        return None
    except json.JSONDecodeError:
        print(
            f"Failed to decode JSON response from API. Response text: {response.text}"
        )
        return None


def make_prediction_commitment(
    stock_ticker, target_date_str, user_prediction, num_seeds
):
    """Creates and saves the commitment for seed generation."""
    timestamp = datetime.datetime.now().isoformat()
    print(f"Seed commitment timestamp: {timestamp}")

    try:
        datetime.datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return False

    event_description = (
        f"Closing price of {stock_ticker} on {target_date_str}. "
        f"User prediction: {user_prediction}"
    )
    print(f"Committed event: {event_description}")

    commitment = {
        "timestamp": timestamp,
        "stock_ticker": stock_ticker,
        "target_date": target_date_str,
        "user_prediction": user_prediction,
        "event_description": event_description,
        "num_seeds": num_seeds,
        "seed_generation_method": "SHA256(timestamp + event_description + actual_stock_price)",
    }

    with open(COMMITMENT_FILE, "w") as f:
        json.dump(commitment, f, indent=4)

    print(f"\nCommitment saved to {COMMITMENT_FILE}")
    print(
        "IMPORTANT: Share this file publicly (e.g., GitHub commit, timestamped service) BEFORE the target date."
    )
    print(
        f"After the target date ({target_date_str}) has passed, run the 'generate' command."
    )
    return True


def generate_seeds_from_commitment(api_key_param):
    """Loads commitment, fetches stock price, and generates seeds."""
    if not os.path.exists(COMMITMENT_FILE):
        print(f"Error: Commitment file '{COMMITMENT_FILE}' not found.")
        print("Please run the 'predict' stage first to create this file.")
        return []

    with open(COMMITMENT_FILE, "r") as f:
        commitment = json.load(f)

    timestamp = commitment["timestamp"]
    stock_ticker = commitment["stock_ticker"]
    target_date_str = commitment["target_date"]
    user_prediction = commitment[
        "user_prediction"
    ]  # This is part of event_description, which is hashed.
    event_description = commitment["event_description"]
    num_seeds = commitment["num_seeds"]

    print(f"\nLoaded commitment from {COMMITMENT_FILE}.")
    print(
        f"Original user prediction for {stock_ticker} on {target_date_str}: '{user_prediction}'"
    )  # Added this line
    print(
        f"Attempting to fetch actual closing price for event: '{event_description}'..."
    )

    api_key = api_key_param or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print(
            "Alpha Vantage API key is required. Provide it via --api-key argument or set ALPHA_VANTAGE_API_KEY environment variable."
        )
        return []

    # Check if target date has passed (optional, but good practice)
    target_date_obj = datetime.datetime.strptime(target_date_str, "%Y-%m-%d").date()
    if datetime.date.today() <= target_date_obj:
        print(
            f"Warning: The target date {target_date_str} has not passed yet or is today."
        )
        # Allow proceeding, but it's unusual if the price isn't final.
        # Consider adding a stricter check or user confirmation if needed.

    actual_closing_price = fetch_stock_price(api_key, stock_ticker, target_date_str)

    if actual_closing_price is None:
        print("Could not retrieve the stock price. Seed generation aborted.")
        print(
            "Please ensure the ticker and date are correct, the market was open, and your API key is valid."
        )
        return []

    print(
        f"Successfully fetched closing price for {stock_ticker} on {target_date_str}: {actual_closing_price}"
    )
    print(
        f"Comparing with original prediction: '{user_prediction}'"
    )  # Added this line for clarity
    event_outcome = str(actual_closing_price)  # Ensure it's a string for hashing

    # Step 5: Generate seeds deterministically
    # The seed base includes the original timestamp, the full event description (which has ticker, date, prediction),
    # and the actual event outcome (stock price).
    seed_base_string = timestamp + event_description + event_outcome
    seed_base = hashlib.sha256(seed_base_string.encode()).hexdigest()
    seeds = []

    for i in range(num_seeds):
        seed_hex = hashlib.sha256((seed_base + str(i)).encode()).hexdigest()
        seed = int(seed_hex[:8], 16)  # Use first 8 hex chars for a 32-bit seed
        seeds.append(seed)

    # Step 6: Save the complete record with seeds
    result = {
        **commitment,  # Spread the original commitment details
        "event_outcome_data": {
            "source": "Alpha Vantage API",
            "fetched_closing_price": actual_closing_price,
        },
        "seeds": seeds,
        "verification_inputs_for_hash": {
            "timestamp": timestamp,
            "event_description": event_description,
            "event_outcome (actual_price)": event_outcome,
        },
        "seed_base_string_hashed": seed_base_string,  # For easier verification
        "final_seed_base_hash": seed_base,
        "verification_date": datetime.datetime.now().isoformat(),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nVerifiable seeds generated and saved to {RESULTS_FILE}:")
    for i, s in enumerate(seeds):
        print(f"Seed {i + 1}: {s}")

    print(
        "\nTo verify: ensure 'seed_commitment.json' (or its public copy) matches the one used for this generation."
    )
    print(
        "Then, using the 'timestamp', 'event_description' from the commitment, and the 'fetched_closing_price',"
    )
    print("recalculate the seed_base and individual seeds. They should match.")

    return seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate verifiable seeds based on future stock prices."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Predict Subcommand ---
    predict_parser = subparsers.add_parser(
        "predict", help="Make a prediction and save the commitment."
    )
    predict_parser.add_argument(
        "--stock-ticker",
        required=True,
        help="Stock ticker symbol (e.g., MSFT, AAPL).",
    )
    predict_parser.add_argument(
        "--target-date",
        required=True,
        help="Future date for the stock's closing price (YYYY-MM-DD).",
    )
    predict_parser.add_argument(
        "--prediction-text",
        required=True,
        help="Your prediction about the stock's closing price (e.g., 'will be above $200').",
    )
    predict_parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds to generate (default: 5).",
    )

    # --- Generate Subcommand ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate seeds based on a saved commitment and actual stock price.",
    )
    generate_parser.add_argument(
        "--api-key",
        help="Alpha Vantage API key. Can also be set via ALPHA_VANTAGE_API_KEY environment variable.",
        default=None,  # Will check env var if not provided
    )

    args = parser.parse_args()

    if args.command == "predict":
        if args.num_seeds <= 0:
            print("Error: Number of seeds must be positive.")
        else:
            make_prediction_commitment(
                args.stock_ticker.upper(),
                args.target_date,
                args.prediction_text,
                args.num_seeds,
            )
    elif args.command == "generate":
        generate_seeds_from_commitment(args.api_key)
