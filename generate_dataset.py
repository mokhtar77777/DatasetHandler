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

    
def apply_score_custom_rules(cfg, engagement_ratio, clicked, like_dislike):
    """
    Applies custom rules (if provided in config) to calculate the user score.

    Args:
        cfg (dict): Loaded YAML config.
        engagement_ratio (float): Ratio of time_spent to summary_length.
        clicked (int): 0 or 1, whether the article was clicked.
        like_dislike (int): -1, 0, or 1 representing user sentiment.

    Returns:
        float: A score sampled from a normal distribution defined in matched rule,
               or a default normal distribution if no rule matches.
    """
    for rule in cfg['score_custom_rules']:
        cond = rule['conditions']
        min_engagement = cond.get('engagement_min', -float('inf'))
        max_engagement = cond.get('engagement_max', float('inf'))

        if not (min_engagement <= engagement_ratio < max_engagement):
            continue

        if cond.get('clicked') is not None and clicked != cond['clicked']:
            continue

        if cond.get('like_dislike') is not None and like_dislike != cond['like_dislike']:
            continue


        loc = rule['output']['loc']
        scale = rule['output']['scale']
        return np.random.normal(loc=loc, scale=scale)

    return np.random.normal(loc=cfg['score_custom_rules_default_value']['loc'], scale=cfg['score_custom_rules_default_value']['scale'])

def apply_like_dislike_custom_rules(cfg, engagement_ratio, clicked):
    """
    Applies custom rules to determine the like/dislike sentiment of the user.

    Args:
        cfg (dict): Loaded YAML config.
        engagement_ratio (float): Ratio of time_spent to summary_length.
        clicked (int): 0 or 1.

    Returns:
        int: -1 for dislike, 0 for neutral, 1 for like based on rule-matched probabilities.
    """
    for i in range(len(cfg['like_dislike']['custom_rules'])):
        cond = cfg['like_dislike']['custom_rules'][i]['conditions']
        min_engagement = cond.get('engagement_min', -float('inf'))
        max_engagement = cond.get('engagement_max', float('inf'))

        if not (min_engagement <= engagement_ratio < max_engagement):
            continue

        if cond.get('clicked') is not None and clicked != cond['clicked']:
            continue


        # All conditions met — return score
        probability = cond['probability']
        return np.random.choice([-1,0,1], p=probability)

    # If no rule matched, return default score or raise warning
    return np.random.choice([-1,0,1], p=cfg['like_dislike']['custom_rules_default_value'])  

def generate_feature(cfg):
    """
    Generates a single synthetic sample of user interaction based on config rules.

    Process includes:
    - Sampling news type
    - Generating summary length and time spent
    - Determining if user clicked
    - Inferring like/dislike sentiment

    Args:
        cfg (dict): Loaded YAML config.

    Returns:
        list: [news_type, summary_length, time_spent, clicked, like_dislike]
    """
    news_types = cfg['news_types']
    news_type = random.choice(news_types)
    summary_length = np.random.randint(cfg['summary_length']['min_summary_length'], cfg['summary_length']['max_summary_length'])
    time_spent = np.random.normal(loc=summary_length * cfg['time_spent']['multiplier'], scale=cfg['time_spent']['std_dev']) 
    time_spent = max(cfg['time_spent']['min_value'], time_spent)
    clicked = np.random.choice([0,1], p=cfg['clicked']['probs_below']) if time_spent < cfg['clicked']['threshold_s'] else np.random.choice([0,1],p=cfg['clicked']['probs_above'])
    
    engagement_ratio = time_spent / summary_length

    if cfg['like_dislike']['rules_type']['hard_coded']:
      if clicked == cfg['like_dislike']['hard_coded_values']['rule_1']['clicked'] and engagement_ratio >= cfg['like_dislike']['hard_coded_values']['rule_1']['engagement']:
        like_dislike = np.random.choice([-1,0,1], p=cfg['like_dislike']['hard_coded_values']['rule_1']['probability'])
      elif clicked == cfg['like_dislike']['hard_coded_values']['rule_2']['clicked'] and engagement_ratio >= cfg['like_dislike']['hard_coded_values']['rule_2']['engagement']:
        like_dislike = np.random.choice([-1,0,1], p=cfg['like_dislike']['hard_coded_values']['rule_2']['probability'])
      elif clicked == cfg['like_dislike']['hard_coded_values']['rule_3']['clicked'] and engagement_ratio < cfg['like_dislike']['hard_coded_values']['rule_3']['engagement']:
        like_dislike = np.random.choice([-1,0,1], p=cfg['like_dislike']['hard_coded_values']['rule_3']['probability'])
      else:
        like_dislike = np.random.choice([-1,0,1], p=cfg['like_dislike']['hard_coded_values']['rule_4']['probability'])
    else:
      like_dislike = apply_like_dislike_custom_rules(cfg, engagement_ratio, clicked)
    
    return [news_type,summary_length,time_spent,clicked,like_dislike]


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
        
    n_samples = cfg['n_samples']
    
    data = []
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        news_type,summary_length,time_spent,clicked,like_dislike = generate_feature(cfg)
        
        engagement_ratio = time_spent / summary_length
        
        if cfg['score_rules_type']['hard_coded']:
            if engagement_ratio >= cfg['score_hard_coded_values']['rule_1']['engagement'] and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_1']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_1']['output']['std'])
            elif engagement_ratio >= cfg['score_hard_coded_values']['rule_2']['engagement'] and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_2']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_2']['output']['std'])
            elif engagement_ratio >= cfg['score_hard_coded_values']['rule_3']['engagement'] and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_3']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_3']['output']['std'])
            elif engagement_ratio >= cfg['score_hard_coded_values']['rule_4']['engagement'] and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_4']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_4']['output']['std'])
            elif engagement_ratio >= cfg['score_hard_coded_values']['rule_5']['engagement'] and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_5']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_5']['output']['std'])
            elif engagement_ratio >= cfg['score_hard_coded_values']['rule_6']['engagement'] and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_6']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_6']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_7']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_7']['engagement_min'] and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_7']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_7']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_8']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_8']['engagement_min'] and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_8']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_8']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_9']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_9']['engagement_min'] and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_9']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_9']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_10']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_10']['engagement_min'] and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_10']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_10']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_11']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_11']['engagement_min'] and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_11']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_11']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_12']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_12']['engagement_min'] and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_12']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_12']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_13']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_13']['engagement_min'] and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_13']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_13']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_14']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_14']['engagement_min'] and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_14']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_14']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_15']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_15']['engagement_min'] and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_15']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_15']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_16']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_16']['engagement_min'] and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_16']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_16']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_17']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_17']['engagement_min'] and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_17']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_17']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_18']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_18']['engagement_min'] and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_18']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_18']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_19']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_19']['engagement_min'] and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_19']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_19']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_20']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_20']['engagement_min'] and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_20']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_20']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_21']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_21']['engagement_min'] and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_21']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_21']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_22']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_22']['engagement_min'] and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_22']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_22']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_23']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_23']['engagement_min'] and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_23']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_23']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_24']['engagement_max'] and engagement_ratio >= cfg['score_hard_coded_values']['rule_24']['engagement_min'] and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_24']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_24']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_25']['engagement'] and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_25']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_25']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_26']['engagement'] and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_26']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_26']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_27']['engagement'] and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_27']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_27']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_28']['engagement'] and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_28']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_28']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_29']['engagement'] and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_29']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_29']['output']['std'])
            elif engagement_ratio < cfg['score_hard_coded_values']['rule_30']['engagement'] and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=cfg['score_hard_coded_values']['rule_30']['output']['mean'], scale=cfg['score_hard_coded_values']['rule_30']['output']['std'])
        else:
            score = apply_score_custom_rules(cfg, engagement_ratio, clicked, like_dislike)

        # Clamp and round
        score = round(max(0.0, min(score, 10.0)), 2)

        data.append([time_spent, clicked, news_type, summary_length, like_dislike, score])
    df = pd.DataFrame(data, columns=['time_spent', 'clicked_full_article', 'news_type', 'summary_length', 'like_dislike', 'user_score'])

    df.to_csv(cfg['file_name'], index=False)

    print("\nUser score summary:")
    print(df['user_score'].describe())