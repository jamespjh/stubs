import argparse
import logging

logger = logging.getLogger(__name__)


def entry():
    logging.basicConfig(level=logging.INFO)
    logger.info('Stubpy called')
    parser = argparse.ArgumentParser(
        description='Echo a message')
    parser.parse_args()
    print(message())


def message():
    return ("Hello from stubpy!")
