"""
News Interaction Data Generator

This script generates a synthetic dataset of user interactions with news articles based on configurable rules.
It uses parameters defined in a YAML configuration file to simulate user behavior like:
- Time spent reading
- Full article button clicked status
- Like/dislike reaction
- Score (user satisfaction/feedback)

The output is saved as a CSV file with features: time_spent, clicked_full_article, news_type, summary_length, like_dislike, user_score.

Configuration should include:
- Rules for generating 'clicked', 'like_dislike', and 'user_score'
- Distributions and probabilities for each rule

Dependencies: pandas, numpy, yaml, tqdm, random
"""


import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import yaml
import time
    
def apply_score_rules(cfg, engagement_ratio, clicked, like_dislike):
    """
    Applies rules to calculate the user score.

    Args:
        cfg (dict): Loaded YAML config.
        engagement_ratio (float): Ratio of time_spent to summary_length.
        clicked (int): 0 or 1, whether the article was clicked.
        like_dislike (int): -1, 0, or 1 representing user sentiment.

    Returns:
        float: A score sampled from a normal distribution defined in matched rule,
               or a default normal distribution if no rule matches.
    """
    for rule in cfg['score_coded_values']:
        cond = rule['condition']
        min_engagement = cond.get('engagement_min', -float('inf'))
        max_engagement = cond.get('engagement_max', float('inf'))

        if not (min_engagement <= engagement_ratio < max_engagement):
            continue

        if cond.get('clicked_value') is not None and clicked != cond['clicked_value']:
            continue

        if cond.get('like_dislike_value') is not None and like_dislike != cond['like_dislike_value']:
            continue


        loc = rule['output']['mean']
        scale = rule['output']['std']
        return np.random.normal(loc=loc, scale=scale)

    return np.random.normal(loc=cfg['score_rules_default_value']['mean'], scale=cfg['score_rules_default_value']['std'])

def apply_like_dislike_rules(cfg, engagement_ratio, clicked):
    """
    Applies rules to determine the like/dislike sentiment of the user.

    Args:
        cfg (dict): Loaded YAML config.
        engagement_ratio (float): Ratio of time_spent to summary_length.
        clicked (int): 0 or 1.

    Returns:
        int: -1 for dislike, 0 for neutral, 1 for like based on rule-matched probabilities.
    """
    for rule in cfg['like_dislike']['coded_values']:
        cond = rule['condition']
        min_engagement = cond.get('engagement_min', -float('inf'))
        max_engagement = cond.get('engagement_max', float('inf'))

        if not (min_engagement <= engagement_ratio < max_engagement):
            continue

        if cond.get('clicked') is not None and clicked != cond['clicked']:
            continue


        # All conditions met — return score
        probability = rule['probability']
        return np.random.choice([-1,0,1], p=probability)

    # If no rule matched, return default score or raise warning
    return np.random.choice([-1,0,1], p=cfg['like_dislike']['like_dislike_rules_default_value'])  

def generate_feature(cfg):
    """
    Generates a single synthetic sample of user interaction based on config rules.

    Process includes:
    - Sampling news type
    - Generating summary length and time spent
    - Determining if user clicked
    - Inferring like/dislike sentiment
    - Calculating user score based on engagement ratio and rules

    Args:
        cfg (dict): Loaded YAML config.

    Returns:
        list: [news_type, summary_length, time_spent, clicked, like_dislike,score]
    """
    n_samples = cfg['n_samples']
    news_types = cfg['news_types']

    # Vectorized precomputations
    summary_lengths = np.random.randint(
        cfg['summary_length']['min_summary_length'],
        cfg['summary_length']['max_summary_length'],
        size=n_samples
    )

    news_type_choices = np.random.choice(news_types, size=n_samples)

    time_spent_raw = np.random.normal(
        loc=summary_lengths * cfg['time_spent_per_summary_len']['multiplier'],
        scale=cfg['time_spent_per_summary_len']['std_dev']
    )
    time_spent_clamped = np.maximum(cfg['time_spent_per_summary_len']['min_value'], time_spent_raw)

    clicked_precomputed = np.where(
        time_spent_clamped < cfg['clicked']['threshold_s'],
        np.random.choice([0, 1], size=n_samples, p=cfg['clicked']['probs_below']),
        np.random.choice([0, 1], size=n_samples, p=cfg['clicked']['probs_above'])
    )
    engagement_ratios = time_spent_clamped / summary_lengths

    data = []
    for i in tqdm(range(n_samples), desc="Generating samples"):
        summary_length = summary_lengths[i]
        news_type = news_type_choices[i]
        time_spent = time_spent_clamped[i]
        clicked_val = clicked_precomputed[i]
        engagement_ratio = engagement_ratios[i]

        # Keep rule-based logic unchanged
        like_dislike = apply_like_dislike_rules(cfg, engagement_ratio, clicked_val)

        score = apply_score_rules(cfg, engagement_ratio, clicked_val, like_dislike)

        score = round(max(0.0, min(score, 10.0)), 2)

        data.append([time_spent, clicked_val, news_type, summary_length, like_dislike, score])
    
    return data


if __name__ == "__main__":
    """
    Main execution block:
    
    1. Loads configuration settings from 'config.yaml', which defines parameters for data generation.
    2. Iterates 'n_samples' times to generate synthetic user interaction data based on rules.
        - Each sample includes news_type, summary_length, time_spent, clicked (0/1), like/dislike.
        - Engagement ratio = time_spent / summary_length is used to determine user behavior.
    3. Applies scoring rules:
        - If 'hard_coded' is True in the config, a fixed set of conditional rules is used to assign a score.
        - Otherwise, user-defined rules in the YAML file are applied to calculate the score.
    4. Scores are sampled from normal distributions based on matched conditions and clamped to [0, 10].
    5. Appends each sample to a dataset list.
    6. Converts the list to a pandas DataFrame and saves it as a CSV file.
    7. Prints a summary of the 'user_score' distribution.
    """
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    start_time = time.time()
    data = generate_feature(cfg)
    end_time = time.time()
    print(f"Data generation completed in {end_time - start_time:.2f} seconds.")
    df = pd.DataFrame(data, columns=['time_spent', 'clicked_full_article', 'news_type', 'summary_length', 'like_dislike', 'user_score'])

    df.to_csv(cfg['file_name'], index=False)

    print("\nUser score summary:")
    print(df['user_score'].describe())