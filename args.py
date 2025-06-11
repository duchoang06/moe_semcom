import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the MOE SemCom model.")
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Pre-trained model name')
    
    args = parser.parse_args()
    
    return args