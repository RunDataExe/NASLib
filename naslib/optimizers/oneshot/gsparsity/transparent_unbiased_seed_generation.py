import hashlib
import datetime
import requests
import json
import os


def generate_verifiable_seeds(num_seeds=5):
    """Generate seeds based on future public data that can't be predicted."""

    # Step 1: Create timestamp of when we're making this commitment
    timestamp = datetime.datetime.now().isoformat()
    print(f"Seed commitment timestamp: {timestamp}")

    # Step 2: Instruct the user to choose a future event
    print("\nTo ensure verifiability, choose a future public event:")
    print("1. Tomorrow's closing price of a specific stock")
    print("2. Hash of a specific future Bitcoin block")
    print("3. Headline of tomorrow's newspaper")

    choice = input("Enter your choice (1-3): ")
    event_description = input(
        "Describe the exact event (e.g., 'AAPL closing price on April 3, 2025'): "
    )

    # Step 3: Create and save commitment file
    commitment = {
        "timestamp": timestamp,
        "future_event": event_description,
        "num_seeds": num_seeds,
    }

    with open("seed_commitment.json", "w") as f:
        json.dump(commitment, f, indent=4)

    print(f"\nCommitment saved to seed_commitment.json")
    print("IMPORTANT: Share this file publicly before continuing (e.g., GitHub commit)")
    input("Press Enter after you've shared the commitment publicly...")

    # Step 4: After the event occurs, get the data
    event_data = input(
        f"\nEnter the actual value of '{event_description}' now that it has occurred: "
    )

    # Step 5: Generate seeds deterministically from the event data
    seed_base = hashlib.sha256(
        (timestamp + event_description + event_data).encode()
    ).hexdigest()
    seeds = []

    for i in range(num_seeds):
        # Generate each seed by hashing the base with an index
        seed_hex = hashlib.sha256((seed_base + str(i)).encode()).hexdigest()
        # Convert first 8 characters of hex to integer (max 32 bits)
        seed = int(seed_hex[:8], 16)
        seeds.append(seed)

    # Step 6: Save the complete record with seeds
    result = {
        **commitment,
        "event_outcome": event_data,
        "seeds": seeds,
        "verification_date": datetime.datetime.now().isoformat(),
    }

    with open("nas_experiment_seeds.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nVerifiable seeds generated and saved to nas_experiment_seeds.json:")
    for i, seed in enumerate(seeds):
        print(f"Seed {i + 1}: {seed}")

    return seeds


if __name__ == "__main__":
    generate_verifiable_seeds()
