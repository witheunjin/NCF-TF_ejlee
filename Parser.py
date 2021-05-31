import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NCF.")
    
    parser.add_argument('--data_size', type=str,default='100K',help='Data Size(ex.NAME(Ratings|Movies|Users)) |__100K(100,000|9,000|600) |__1M(1,000,000|4,000|6,000) |__20M(20,000,000|27,000|138,000) |__25M(25,000,000|62,000|162,000) |__27M(27,000,000|58,000|280,000)')
    return parser.parse_args()
