import argparse
import logging

logger = logging.getLogger(__name__)

def entry():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Demonstrate GPU speedup')
    parser.parse_args()
    print(message())

def message():
    return ("Nothing yet")
