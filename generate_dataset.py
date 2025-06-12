import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    
def apply_custom_rules(cfg, engagement_ratio, clicked, like_dislike):
    for rule in cfg['custom_rules']:
        cond = rule['conditions']
        min_engagement = cond.get('engagement_min', -float('inf'))
        max_engagement = cond.get('engagement_max', float('inf'))

        if not (min_engagement <= engagement_ratio < max_engagement):
            continue

        if cond.get('clicked') is not None and clicked != cond['clicked']:
            continue

        if cond.get('like_dislike') is not None and like_dislike != cond['like_dislike']:
            continue

        # All conditions met — return score
        loc = rule['output']['loc']
        scale = rule['output']['scale']
        return np.random.normal(loc=loc, scale=scale)

    return np.random.normal(loc=5.0, scale=1.0)

    

def generate_feature(cfg):
    news_types = cfg['news_types']
    news_type = random.choice(news_types)
    summary_length = np.random.randint(cfg['summary_length']['min_summary_length'], cfg['summary_length']['max_summary_length']) 
    time_spent = np.random.normal(loc=summary_length * cfg['time_spent']['multiplier'], scale=cfg['time_spent']['std_dev']) #in ms
    time_spent = max(cfg['time_spent']['min_value'], time_spent) 

    clicked = np.random.choice(cfg['clicked']['choice'], p=cfg['clicked']['probs_below']) if time_spent < cfg['clicked']['threshold_ms'] else np.random.choice(cfg['clicked']['choice'],p=cfg['clicked']['probs_above'])
    like_dislike_vals = list(map(int, cfg['like_dislike_probs'].keys()))
    like_dislike_probs = list(cfg['like_dislike_probs'].values())
    like_dislike = np.random.choice(like_dislike_vals, p=like_dislike_probs)
    
    return [news_type,summary_length,time_spent,clicked,like_dislike]


if __name__ == "__main__":
    
    n_samples = cfg['n_samples']
    
    data = []
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        news_type,summary_length,time_spent,clicked,like_dislike = generate_feature(cfg)
        
        engagement_ratio = time_spent / summary_length
        
        if cfg['rules_type']['hard_coded']:
            if engagement_ratio >= 110 and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=9.5, scale=0.5)
            elif engagement_ratio >= 110 and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=8.0, scale=0.5)
            elif engagement_ratio >= 110 and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=6.0, scale=0.5)
            elif engagement_ratio >= 110 and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=8.5, scale=0.5) 
            elif engagement_ratio >= 110 and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=7.0, scale=0.5)
            elif engagement_ratio >= 110 and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=5.0, scale=0.5)
            elif engagement_ratio < 120 and engagement_ratio >= 100 and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=8.5, scale=0.5)     
            elif engagement_ratio < 120 and engagement_ratio >= 100 and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=7.0, scale=0.5)
            elif engagement_ratio < 120 and engagement_ratio >= 100 and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=5.0, scale=0.5)
            elif engagement_ratio < 120 and engagement_ratio >= 100 and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=7.5, scale=0.5)
            elif engagement_ratio < 120 and engagement_ratio >= 100 and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=6.0, scale=0.5)
            elif engagement_ratio < 120 and engagement_ratio >= 100 and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=4.0, scale=0.5)
            elif engagement_ratio < 100 and engagement_ratio >= 95 and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=7.5, scale=0.5)
            elif engagement_ratio < 100 and engagement_ratio >= 95 and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=6.0, scale=0.5)
            elif engagement_ratio < 100 and engagement_ratio >= 95 and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=4.0, scale=0.5)
            elif engagement_ratio < 100 and engagement_ratio >= 95 and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=6.5, scale=0.5)
            elif engagement_ratio < 100 and engagement_ratio >= 95 and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=5.0, scale=0.5)
            elif engagement_ratio < 100 and engagement_ratio >= 95 and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=3.0, scale=0.5)
            elif engagement_ratio < 95 and engagement_ratio >= 85 and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=6.5, scale=0.6)
            elif engagement_ratio < 95 and engagement_ratio >= 85 and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=5.0, scale=0.6)
            elif engagement_ratio < 95 and engagement_ratio >= 85 and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=3.0, scale=0.6)
            elif engagement_ratio < 95 and engagement_ratio >= 85 and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=5.5, scale=0.6)
            elif engagement_ratio < 95 and engagement_ratio >= 85 and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=4.0, scale=0.6)
            elif engagement_ratio < 95 and engagement_ratio >= 85 and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=2.0, scale=0.6)
            elif engagement_ratio < 85 and clicked == 1 and like_dislike == 1:
                score = np.random.normal(loc=5.5, scale=0.6)
            elif engagement_ratio < 85 and clicked == 1 and like_dislike == 0:
                score = np.random.normal(loc=4.0, scale=0.6)
            elif engagement_ratio < 85 and clicked == 1 and like_dislike == -1:
                score = np.random.normal(loc=2.0, scale=0.6)
            elif engagement_ratio < 85 and clicked == 0 and like_dislike == 1:
                score = np.random.normal(loc=4.5, scale=0.6)
            elif engagement_ratio < 85 and clicked == 0 and like_dislike == 0:
                score = np.random.normal(loc=3.0, scale=0.6)
            elif engagement_ratio < 85 and clicked == 0 and like_dislike == -1:
                score = np.random.normal(loc=1.0, scale=0.6)
        elif cfg['rules_type']['custom']:
            score = apply_custom_rules(cfg, engagement_ratio, clicked, like_dislike)

        # Clamp and round
        score = round(max(0.0, min(score, 10.0)), 2)

        data.append([time_spent, clicked, news_type, summary_length, like_dislike, score])
    df = pd.DataFrame(data, columns=['time_spent', 'clicked_full_article', 'news_type', 'summary_length', 'like_dislike', 'user_score'])

    df.to_csv(cfg['file_name'], index=False)

    print("\nUser score summary:")
    print(df['user_score'].describe())