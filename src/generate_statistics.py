"""
News Interaction Data Generator

This script takes the synthetic dataset of user interactions with news articles and generates statistics from it.
It uses parameters defined in a YAML configuration file to simulate user behavior like:
- Time spent reading
- Full article button clicked status
- Like/dislike reaction
- Score (user satisfaction/feedback)

The output is saved as a CSV file with features: time_spent, clicked_full_article, news_type, summary_length, like_dislike, user_score.

Configuration should include:
- Rules for generating 'clicked', 'like_dislike', and 'user_score'
- Distributions and probabilities for each rule

Dependencies: pandas, numpy, yaml, tqdm, os, argparse
"""

import yaml
import pandas as pd
import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--configfile", required=True, help="Path of .YAML configuration file")
    args = argparser.parse_args()

    # Open configuration file
    with open(args.configfile, "r") as f:
        cfg = yaml.safe_load(f)

    # Read dataset
    print(pd.read_csv(cfg["dataset_destination"]))
