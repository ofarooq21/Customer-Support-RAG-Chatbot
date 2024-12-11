# evaluate_ragas.py

import os
import json
import asyncio
from ragas.metrics import ResponseRelevancy, Faithfulness
from ragas import SingleTurnSample


# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'  

def load_interactions(file_path):
    interactions = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                interaction = json.loads(line)
                # Validate interaction structure
                if all(key in interaction for key in ("query", "context", "response")):
                    interactions.append(interaction)
                else:
                    print("Skipping malformed interaction:", interaction)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
    return interactions

async def main():
    interactions = load_interactions('interaction_logs.jsonl')

    # Initialize the metrics
    response_relevancy_metric = ResponseRelevancy()
    faithfulness_metric = Faithfulness()

    # Evaluate each interaction asynchronously
    response_relevancy_scores = []
    faithfulness_scores = []

    for interaction in interactions:
        query = interaction['query']
        context = interaction['context']
        response = interaction['response']

        # Create SingleTurnSample object
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=[context]
        )

        try:
            # ResponseRelevancy
            score = await response_relevancy_metric.single_turn_ascore(sample)
            response_relevancy_scores.append(score)
        except Exception as e:
            print(f"Error evaluating ResponseRelevancy: {e}")
            response_relevancy_scores.append(None)

        try:
            # Faithfulness
            score = await faithfulness_metric.single_turn_ascore(sample)
            faithfulness_scores.append(score)
        except Exception as e:
            print(f"Error evaluating Faithfulness: {e}")
            faithfulness_scores.append(None)

    # Compute average scores
    def compute_average(scores):
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            return sum(valid_scores) / len(valid_scores)
        else:
            return None

    avg_response_relevancy = compute_average(response_relevancy_scores)
    avg_faithfulness = compute_average(faithfulness_scores)

    print("\nRAGAS Evaluation Results:")
    print(f"Average ResponseRelevancy Score: {avg_response_relevancy}")
    print(f"Average Faithfulness Score: {avg_faithfulness}")

if __name__ == "__main__":
    asyncio.run(main())
